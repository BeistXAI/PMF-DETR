"""Configuration, reproducibility and state-dict checkpoint helpers."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Model configuration must contain a model mapping.")
    return deepcopy(config)


def select_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("This implementation supports a single CPU or CUDA device.")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a compatible PyTorch CUDA wheel or use --device cpu.")
    return selected


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def capture_rng_state(generator=None):
    numpy_state = np.random.get_state()
    result = {"python": random.getstate(), "torch": torch.get_rng_state(),
              "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                        "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gauss": numpy_state[4]}}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        result["loader"] = generator.get_state()
    return result


def restore_rng_state(state, generator=None):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gauss"]))
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint; exact RNG restoration is unavailable.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if generator is not None and "loader" in state:
        generator.set_state(state["loader"].cpu())


def atomic_save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    """Load tensors and primitive metadata only; never unpickle a serialized model object."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Expected a PMF-DETR format_version=1 state-dict checkpoint.")
    if not isinstance(checkpoint.get("model"), dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Checkpoint is missing model state or configuration.")
    return checkpoint


def load_model_checkpoint(path, device="cpu"):
    from .model import build_model

    checkpoint = load_checkpoint(path)
    selected = select_device(device)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(selected).eval()
    return model, checkpoint


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
