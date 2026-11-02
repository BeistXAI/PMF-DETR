"""Create reproducible, patient-disjoint image lists without moving image data.

Manifest CSV: image,patient_id. Image paths are relative to --root and must contain
an images/ component so the corresponding YOLO label path is unambiguous.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def split_manifest(manifest, root, output, ratios=(0.7, 0.1, 0.2), seed=42):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Train/val/test ratios must be nonnegative and sum to 1.")
    grouped = defaultdict(list)
    seen = set()
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "patient_id"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires image,patient_id columns.")
        for row in reader:
            patient = row["patient_id"].strip()
            if not patient:
                raise ValueError("Every image must have a nonempty patient_id.")
            image = (root / row["image"]).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError(f"Image must exist inside --root: {image}")
            if image in seen:
                raise ValueError(f"Duplicate image in manifest: {image}")
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError(f"Expected an images/ path: {image}")
            index = len(parts) - 1 - parts[::-1].index("images")
            parts[index] = "labels"
            if not Path(*parts).with_suffix(".txt").is_file():
                raise ValueError(f"Missing YOLO label for {image}; negatives need an empty file.")
            seen.add(image)
            grouped[patient].append(image.relative_to(root).as_posix())
    patients = sorted(grouped)
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if len(patients) < len(active):
        raise ValueError("Not enough distinct patients for the requested nonempty splits.")
    random.Random(seed).shuffle(patients)
    # Allocate patient counts by the largest-remainder method. Ratios refer to
    # patients, not slices: one patient may contribute many more images.
    ideals = [len(patients) * r for r in ratios]
    counts = [int(x) for x in ideals]
    residual = len(patients) - sum(counts)
    order = sorted(active, key=lambda i: (ideals[i] - counts[i], ratios[i]), reverse=True)
    for index in order[:residual]:
        counts[index] += 1
    for index in active:
        if counts[index] == 0:
            donor = max(active, key=lambda j: counts[j])
            if counts[donor] <= 1:
                raise ValueError("Unable to allocate nonempty splits.")
            counts[donor] -= 1
            counts[index] = 1
    names = ("train", "val", "test")
    expected = [output / f"{name}.txt" for name in names] + [output / "split_summary.json", output / "assignments.csv"]
    if any(path.exists() for path in expected):
        raise FileExistsError("Output already has split files; choose a new directory to preserve the existing split.")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "ratios": list(ratios), "ratio_unit": "patients", "root": str(root), "splits": {}}
    assignment = []
    offset = 0
    for name, count in zip(names, counts):
        members = patients[offset:offset + count]
        offset += count
        images = sorted(image for patient in members for image in grouped[patient])
        (output / f"{name}.txt").write_text("".join(f"{image}\n" for image in images), encoding="utf-8")
        summary["splits"][name] = {"patients": len(members), "images": len(images)}
        assignment.extend({"image": image, "patient_id": patient, "split": name} for patient in members for image in grouped[patient])
    with (output / "assignments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "patient_id", "split"])
        writer.writeheader()
        writer.writerows(assignment)
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.1, 0.2], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(split_manifest(args.manifest, args.root, args.output, args.ratios, args.seed), indent=2))


if __name__ == "__main__":
    main()

