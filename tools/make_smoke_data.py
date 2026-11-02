"""Generate tiny artificial images for software checks, not medical evaluation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def make_smoke_data(output, seed=7):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory for synthetic test data.")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for split, number in (("train", 6), ("val", 4), ("test", 2)):
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True, exist_ok=True)
        for index in range(number):
            width, height = (112, 80) if index % 2 else (96, 96)
            pixels = rng.integers(0, 35, (height, width), dtype=np.uint8)
            image = Image.fromarray(pixels)
            lines = []
            if index % 3 != 0:
                x1, y1, x2, y2 = int(width * .2), int(height * .25), int(width * .7), int(height * .6)
                ImageDraw.Draw(image).rectangle([x1, y1, x2-1, y2-1], fill=210)
                box = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
                lines.append("0 " + " ".join(f"{value:.9f}" for value in box))
            stem = f"{split}_{index:03d}"
            relative = Path("images") / split / f"{stem}.png"
            image.save(output / relative)
            (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            records.append({"image": relative.as_posix(), "patient_id": f"synthetic_{split}_{index}"})
    dataset = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": ["fracture"]}
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    with (output / "patients.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "patient_id"])
        writer.writeheader()
        writer.writerows(records)
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pmf_detr_r18.yaml").read_text())
    config["description"] = "Tiny synthetic software smoke test; NOT a paper experiment or clinical benchmark."
    config["model"].update(hidden_dim=32, num_queries=20, num_decoder_layers=1, num_heads=4, num_points=2, dim_feedforward=64, num_denoising=8, fpn_depth=1)
    config["train"].update(imgsz=64, epochs=1, batch_size=2, workers=0, seed=seed, warmup_epochs=0)
    config["eval"]["max_dets"] = 20
    (output / "smoke_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(make_smoke_data(args.output, args.seed))


if __name__ == "__main__":
    main()

