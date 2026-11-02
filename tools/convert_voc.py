"""Convert LabelImg/Pascal VOC boxes to YOLO labels without moving images.

Default VOC coordinates are inclusive and 1-based. Use --zero-based only when
your export explicitly uses zero-based half-open xyxy coordinates.
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_voc(annotations, output, classes=("fracture",), zero_based=False):
    classes = list(classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique and nonempty.")
    files = sorted(Path(annotations).glob("*.xml"))
    if not files:
        raise FileNotFoundError(f"No XML annotations found in {annotations}")
    output = Path(output)
    staged = []
    for file in files:
        tree = ET.parse(file).getroot()
        width = int(tree.findtext("size/width", "0"))
        height = int(tree.findtext("size/height", "0"))
        if min(width, height) <= 0:
            raise ValueError(f"Invalid image dimensions: {file}")
        lines = []
        for obj in tree.findall("object"):
            name = obj.findtext("name", "").strip()
            if name not in classes:
                raise ValueError(f"Unknown class {name!r} in {file}; provide --classes in the intended class order.")
            coords = [float(obj.findtext(f"bndbox/{key}", "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
            if not all(math.isfinite(x) for x in coords):
                raise ValueError(f"Non-finite bounding box in {file}")
            x1, y1, x2, y2 = coords
            if not zero_based:
                x1 -= 1
                y1 -= 1
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid or out-of-bounds box {coords} in {file}")
            box = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
            lines.append(f"{classes.index(name)} " + " ".join(f"{v:.9f}" for v in box))
        target = output / file.with_suffix(".txt").name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing label: {target}")
        staged.append((target, "\n".join(lines) + ("\n" if lines else "")))
    output.mkdir(parents=True, exist_ok=True)
    for target, text in staged:
        target.write_text(text, encoding="utf-8")
    return len(staged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", nargs="+", default=["fracture"])
    parser.add_argument("--zero-based", action="store_true")
    args = parser.parse_args()
    print(f"Converted {convert_voc(args.annotations, args.output, args.classes, args.zero_based)} annotation files.")


if __name__ == "__main__":
    main()
