from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml

from .data import YoloDetectionDataset, collate_fn, load_data_config, validate_split_separation
from .metrics import DetectionEvaluator, postprocess
from .utils import (atomic_save_checkpoint, capture_rng_state, load_checkpoint, restore_rng_state,
                    seed_everything, seed_worker, select_device, write_json)


TRAIN_DEFAULTS = {"epochs": 300, "batch_size": 4, "imgsz": 640, "workers": 4,
                  "lr": 1e-4, "backbone_lr": 1e-5, "weight_decay": 1e-4,
                  "clip_grad_norm": 0.1, "warmup_epochs": 3, "min_lr_ratio": 0.01,
                  "hflip": 0.5, "seed": 42, "amp": False}
EVAL_DEFAULTS = {"score_threshold": 0.001, "conf": 0.25, "max_dets": 300}


def make_loader(dataset, batch_size, workers=0, shuffle=False, generator=None, device="cpu"):
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers must be nonnegative.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate_fn, pin_memory=str(device).startswith("cuda"),
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)


def _targets_to_device(targets, device):
    return [{"labels": t["labels"].to(device, non_blocking=True),
             "boxes": t["boxes"].to(device, non_blocking=True)} for t in targets]


@torch.no_grad()
def evaluate(model, loader, device="cpu", num_classes=1, names=None, conf=0.25,
             score_threshold=0.001, max_dets=300):
    """AP uses low-threshold predictions; P/R/F1 use conf independently at IoU=0.5."""
    if score_threshold > conf:
        raise ValueError("AP score_threshold must be <= the P/R/F1 confidence threshold.")
    evaluator = DetectionEvaluator(num_classes, conf=conf, names=names)
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        for images, targets in loader:
            outputs = model(images.to(device, non_blocking=True))
            evaluator.update(postprocess(outputs, max_dets=max_dets, score_threshold=score_threshold), targets)
    finally:
        model.train(was_training)
    metrics = evaluator.compute()
    metrics.update(score_threshold=score_threshold, max_dets=max_dets,
                   evaluation_seconds=time.perf_counter() - started)
    return metrics


def _dataset_signature(train_dataset, val_dataset):
    """Fingerprint class order, split paths and annotation bytes (not image pixels)."""
    digest = hashlib.sha256(json.dumps(train_dataset.names, ensure_ascii=False).encode())
    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        digest.update(split.encode())
        for image, annotation in zip(dataset.paths, dataset.label_paths):
            image_bytes = str(image).encode()
            label_bytes = annotation.read_bytes()
            digest.update(len(image_bytes).to_bytes(8, "little"))
            digest.update(image_bytes)
            digest.update(len(label_bytes).to_bytes(8, "little"))
            digest.update(label_bytes)
    return digest.hexdigest()


def _lr_factor(epoch, settings):
    warmup = settings["warmup_epochs"]
    if warmup and epoch < warmup:
        return (epoch + 1) / warmup
    duration = max(settings["epochs"] - warmup - 1, 1)
    progress = min(max((epoch - warmup) / duration, 0), 1)
    return settings["min_lr_ratio"] + (1 - settings["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2


def train_model(config, data_yaml, output=None, device="cpu", resume=None, weights=None):
    from .criterion import build_criterion
    from .model import build_model

    if resume and weights:
        raise ValueError("Use --resume or --weights, not both.")
    resume_directory = Path(resume).expanduser().resolve().parent if resume else None
    output = Path(output or resume_directory or "runs/train").expanduser().resolve()
    if resume_directory is not None and output != resume_directory:
        raise ValueError("--resume must continue in the checkpoint's original output directory so best.pt/history stay consistent. Use --weights for a new output experiment.")
    config = deepcopy(config)
    config["train"] = settings = {**TRAIN_DEFAULTS, **config.get("train", {})}
    config["eval"] = eval_settings = {**EVAL_DEFAULTS, **config.get("eval", {})}
    if settings["epochs"] < 1 or settings["lr"] <= 0 or settings["backbone_lr"] <= 0:
        raise ValueError("epochs and learning rates must be positive.")
    if settings["warmup_epochs"] < 0 or settings["clip_grad_norm"] <= 0 or not 0 <= settings["min_lr_ratio"] <= 1:
        raise ValueError("Invalid warmup, gradient clipping, or minimum LR ratio.")
    selected = select_device(device)
    if settings["amp"] and selected.type != "cuda":
        raise ValueError("--amp requires CUDA; use ordinary float32 training on CPU.")
    seed_everything(settings["seed"])
    data_config = load_data_config(data_yaml)
    split_counts = validate_split_separation(data_config)
    num_classes = len(data_config["names"])
    if int(config["model"].get("num_classes", num_classes)) != num_classes:
        raise ValueError("Model num_classes must match dataset names. Edit the model configuration explicitly.")
    config["model"]["num_classes"] = num_classes
    train_dataset = YoloDetectionDataset(data_config, "train", settings["imgsz"], True, settings["hflip"])
    val_dataset = YoloDetectionDataset(data_config, "val", settings["imgsz"])
    signature = _dataset_signature(train_dataset, val_dataset)
    generator = torch.Generator().manual_seed(settings["seed"])
    val_generator = torch.Generator().manual_seed(settings["seed"] + 1)
    train_loader = make_loader(train_dataset, settings["batch_size"], settings["workers"], True, generator, selected)
    val_loader = make_loader(val_dataset, settings["batch_size"], settings["workers"], False, val_generator, selected)
    if (output / "last.pt").exists() and not resume:
        raise FileExistsError(f"{output}/last.pt already exists. Use a new --output or resume explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(config).to(selected)
    criterion = build_criterion(config, num_classes).to(selected)
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith("backbone.") else other).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": settings["backbone_lr"]})
    if other:
        groups.append({"params": other, "lr": settings["lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: _lr_factor(epoch, settings))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])
    else:  # PyTorch 2.0 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=settings["amp"])
    start_epoch, best_map, history = 0, -1.0, []
    if weights:
        initial = load_checkpoint(weights)
        if initial.get("names") != data_config["names"]:
            raise ValueError("Initial checkpoint class names differ from the dataset.")
        model.load_state_dict(initial["model"], strict=True)
    if resume:
        checkpoint = load_checkpoint(resume)
        if checkpoint.get("names") != data_config["names"] or checkpoint.get("dataset_signature") != signature:
            raise ValueError("Resume class names, train/validation image paths, or annotation contents differ from the checkpoint.")
        old_config = checkpoint["config"]
        for section in ("model", "loss", "eval"):
            if old_config.get(section, {}) != config.get(section, {}):
                raise ValueError(f"Resume {section} configuration differs from checkpoint.")
        for key in TRAIN_DEFAULTS:
            if key != "epochs" and old_config["train"][key] != settings[key]:
                raise ValueError(f"Resume train.{key} differs from checkpoint; use --weights for a new experiment.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map, history = float(checkpoint["best_map"]), checkpoint.get("history", [])
        restore_rng_state(checkpoint["rng"], generator)
        if start_epoch >= settings["epochs"]:
            raise ValueError(f"Checkpoint already completed {start_epoch} epochs; increase --epochs to continue.")
        if old_config["train"]["epochs"] != settings["epochs"]:
            print("Resume extends/changes the epoch budget: cosine LR schedule uses the new horizon; this is not an identical uninterrupted run.", flush=True)
            new_lrs = [base_lr * _lr_factor(start_epoch, settings) for base_lr in scheduler.base_lrs]
            for group, new_lr in zip(optimizer.param_groups, new_lrs):
                group["lr"] = new_lr
            scheduler._last_lr = new_lrs
    metadata = {"config": config, "dataset": data_config, "split_counts": split_counts,
                "device": str(selected), "torch_version": str(torch.__version__),
                "parameters": sum(p.numel() for p in model.parameters()),
                "dataset_signature": signature,
                "dataset_signature_scope": "class order, split image paths, and label bytes; excludes image pixel contents"}
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(metadata, output / "experiment.json")
    print(json.dumps({"event": "training_start", "output": str(output), **metadata}, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        accumulated, count = {}, 0
        started = time.perf_counter()
        used_lrs = [group["lr"] for group in optimizer.param_groups]
        for images, targets in train_loader:
            images = images.to(selected, non_blocking=True)
            targets = _targets_to_device(targets, selected)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=selected.type, dtype=torch.float16, enabled=settings["amp"]):
                outputs = model(images, targets)
                losses = criterion(outputs, targets)
                total = losses["loss_total"]
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {total.item()}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.shape[0]
            count += batch_size
            for key, value in losses.items():
                accumulated[key] = accumulated.get(key, 0.0) + float(value.detach()) * batch_size
        metrics = evaluate(model, val_loader, selected, num_classes, data_config["names"], **eval_settings)
        record = {"epoch": epoch + 1, "lr": used_lrs, "epoch_seconds": time.perf_counter() - started,
                  "loss": {key: value / count for key, value in accumulated.items()}, "val": metrics}
        history.append(record)
        improved = metrics["map50_95"] > best_map
        best_map = max(best_map, metrics["map50_95"])
        scheduler.step()
        checkpoint = {"format_version": 1, "model": model.state_dict(), "config": config,
                      "names": data_config["names"], "dataset_signature": signature,
                      "epoch": epoch, "best_map": best_map, "history": history,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "rng": capture_rng_state(generator)}
        if improved:
            atomic_save_checkpoint(checkpoint, output / "best.pt")
        atomic_save_checkpoint(checkpoint, output / "last.pt")
        write_json(history, output / "history.json")
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    return {"output": str(output), "epochs_completed": settings["epochs"], "best_map50_95": best_map,
            "last_checkpoint": str(output / "last.pt"), "best_checkpoint": str(output / "best.pt")}
