"""Check YOLO labels, readable images, split overlap and optional patient leakage."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from pmf_detr.data import load_data_config, resolve_split, validate_split_separation, label_path_for_image, read_yolo_labels


def check_dataset(data, patients=None):
    config = load_data_config(data)
    split_names = [name for name in ("train", "val", "test") if config.get(name)]
    if "train" not in split_names or "val" not in split_names:
        raise ValueError("A training dataset must have distinct train and val splits.")
    validate_split_separation(config, splits=tuple(split_names))
    patient_map = {}
    root = Path(config["path"])
    if patients:
        with Path(patients).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"image", "patient_id"} <= set(reader.fieldnames or []):
                raise ValueError("Patient manifest needs image,patient_id columns.")
            for row in reader:
                path = (root / row["image"]).resolve()
                patient = row["patient_id"].strip()
                if not patient or path in patient_map:
                    raise ValueError(f"Duplicate image or blank patient ID in manifest: {path}")
                patient_map[path] = patient
    report = {"data": str(Path(data).resolve()), "names": config["names"], "patient_groups_checked": bool(patients), "splits": {}}
    patient_split = {}
    for split in split_names:
        counts = Counter()
        members = set()
        images = resolve_split(config, split)
        if not images:
            raise ValueError(f"Empty {split} split.")
        negatives = 0
        for image in images:
            with Image.open(image) as decoded:
                decoded.verify()
            labels, boxes = read_yolo_labels(label_path_for_image(image), config["nc"])
            negatives += int(len(labels) == 0)
            counts.update(int(x) for x in labels)
            if patients:
                if image not in patient_map:
                    raise ValueError(f"Image missing from patient manifest: {image}")
                patient = patient_map[image]
                if patient in patient_split and patient_split[patient] != split:
                    raise ValueError(f"Patient {patient!r} appears in both {patient_split[patient]} and {split}.")
                patient_split[patient] = split
                members.add(patient)
        report["splits"][split] = {"images": len(images), "negative_images": negatives, "boxes": sum(counts.values()), "boxes_by_class": dict(counts)}
        if patients:
            report["splits"][split]["patients"] = len(members)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--patients", help="CSV image,patient_id; image paths relative to dataset root")
    parser.add_argument("--output", help="Optional JSON report")
    args = parser.parse_args()
    report = json.dumps(check_dataset(args.data, args.patients), indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

