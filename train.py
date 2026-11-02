#!/usr/bin/env python3
"""Train PMF-DETR from a YAML configuration and a strictly annotated YOLO dataset."""

import argparse
import json
import torch

from pmf_detr.engine import train_model
from pmf_detr.utils import load_checkpoint, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Model YAML (default: configs/pmf_detr_r18.yaml; resume uses checkpoint configuration)")
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Experiment directory (default: runs/train, or the resume checkpoint's directory)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--threads", type=int, help="PyTorch CPU threads")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable CUDA float16 mixed precision")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--resume", help="Restore model, optimizer, scheduler, scaler, RNG, and completed epoch")
    initial.add_argument("--weights", help="Initialize model weights only for a new experiment")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be positive.")
        torch.set_num_threads(args.threads)
    if args.config:
        config = load_config(args.config)
    elif args.resume:
        config = load_checkpoint(args.resume)["config"]
    else:
        config = load_config("configs/pmf_detr_r18.yaml")
    config.setdefault("train", {})
    for key in ("epochs", "batch_size", "workers", "imgsz", "seed", "lr", "backbone_lr", "amp"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    result = train_model(config, args.data, args.output, args.device, args.resume, args.weights)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
