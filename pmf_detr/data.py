"""Strict YOLO-format slice datasets; no implicit CT windowing or data downloads."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_data_config(path):
    """Resolve ``path`` against the YAML directory and split lists against that root."""
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping.")
    names = config.get("names")
    if isinstance(names, dict):
        try:
            converted = {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Class-name keys must be consecutive integer IDs.") from exc
        if sorted(converted) != list(range(len(converted))):
            raise ValueError("Class IDs in names must start at zero and be consecutive.")
        names = [converted[i] for i in range(len(converted))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Dataset YAML requires a non-empty names list or zero-indexed mapping.")
    if len(set(names)) != len(names):
        raise ValueError("Dataset class names must be unique.")
    if "nc" in config and int(config["nc"]) != len(names):
        raise ValueError("Dataset nc disagrees with the number of class names.")
    root = Path(config.get("path", ".")).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    config.update(path=str(root.resolve()), names=names, nc=len(names), yaml_path=str(yaml_path))
    return config


def resolve_split(config, split):
    """Return unique image paths; text-file entries are relative to dataset ``path``."""
    if split not in config or not config[split]:
        raise ValueError(f"Dataset YAML does not define split {split!r}.")
    root = Path(config["path"])
    sources = config[split] if isinstance(config[split], list) else [config[split]]
    images = []
    for source in sources:
        source = Path(source).expanduser()
        if not source.is_absolute():
            source = root / source
        if source.is_dir():
            images.extend(sorted(p.resolve() for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                image = Path(line).expanduser()
                if not image.is_absolute():
                    image = root / image
                images.append(image.resolve())
        else:
            raise FileNotFoundError(f"Split {split!r} must be an image directory or image-list .txt file: {source}")
    if not images:
        raise ValueError(f"Split {split!r} contains no supported images.")
    if len(images) != len(set(images)):
        raise ValueError(f"Split {split!r} contains duplicate image paths.")
    for image in images:
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"Unsupported or missing image in split {split!r}: {image}")
    return images


def validate_split_separation(config, splits=("train", "val", "test")):
    """Reject path overlap. Patient-level grouping still requires a patient manifest."""
    sets = {split: set(resolve_split(config, split)) for split in splits if config.get(split)}
    keys = list(sets)
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            overlap = sets[first] & sets[second]
            if overlap:
                example = sorted(map(str, overlap))[0]
                raise ValueError(f"Data leakage: {len(overlap)} image(s) shared by {first}/{second}; e.g. {example}")
    return {split: len(paths) for split, paths in sets.items()}


def label_path_for_image(image_path):
    """Replace the final ``images`` path component with ``labels``."""
    path = Path(image_path)
    parts = list(path.parts)
    positions = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if not positions:
        raise ValueError(f"Image path must contain an images/ directory to locate YOLO labels: {path}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(path, num_classes):
    """Read class cx cy w h. Empty files are negatives; missing files are errors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation {path}. Create an empty .txt file for a verified negative image.")
    labels, boxes = [], []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{lineno}: expected class cx cy width height.")
        try:
            cls, cx, cy, width, height = map(float, values)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: non-numeric annotation.") from exc
        if not np.isfinite([cls, cx, cy, width, height]).all():
            raise ValueError(f"{path}:{lineno}: annotation contains NaN or infinity.")
        if cls != int(cls) or not 0 <= int(cls) < num_classes:
            raise ValueError(f"{path}:{lineno}: invalid class ID {cls}; expected [0,{num_classes - 1}].")
        if width <= 0 or height <= 0 or width > 1 or height > 1:
            raise ValueError(f"{path}:{lineno}: width and height must be in (0,1].")
        if min(cx - width / 2, cy - height / 2) < -1e-5 or max(cx + width / 2, cy + height / 2) > 1 + 1e-5:
            raise ValueError(f"{path}:{lineno}: normalized box extends outside the image.")
        labels.append(int(cls))
        boxes.append([cx, cy, width, height])
    return torch.tensor(labels, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def load_image_tensor(path, image_size):
    """Read a pre-windowed 8-bit image, replicate grayscale, stretch to a square."""
    with Image.open(path) as source:
        if source.mode in {"I", "F", "I;16", "I;16B", "I;16L", "I;16N"}:
            raise ValueError(f"{path}: high-bit-depth/float image requires explicit CT windowing before use.")
        original_size = source.size  # width, height
        rgb = source.convert("RGB")
        resized = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous(), original_size


class YoloDetectionDataset(Dataset):
    """One 2-D slice per sample; targets use normalized cxcywh boxes."""

    def __init__(self, config, split="train", image_size=640, augment=False, hflip=0.5):
        self.config = load_data_config(config) if isinstance(config, (str, Path)) else config
        if image_size <= 0 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32.")
        if not 0 <= hflip <= 1:
            raise ValueError("hflip must be in [0,1].")
        self.paths = resolve_split(self.config, split)
        self.label_paths = [label_path_for_image(p) for p in self.paths]
        missing = next((p for p in self.label_paths if not p.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"Missing annotation {missing}. Verified negatives require an empty label file.")
        self.image_size, self.augment, self.hflip = image_size, augment, hflip
        self.names = self.config["names"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image, original_size = load_image_tensor(self.paths[index], self.image_size)
        labels, boxes = read_yolo_labels(self.label_paths[index], len(self.names))
        if self.augment and random.random() < self.hflip:
            image = image.flip(-1)
            boxes[:, 0] = 1.0 - boxes[:, 0]
        return image, {"labels": labels, "boxes": boxes, "image_id": index,
                       "path": str(self.paths[index]), "original_size": original_size}


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
