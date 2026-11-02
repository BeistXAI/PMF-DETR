#!/usr/bin/env python3
"""Save pixel-coordinate detections and annotated images from a PMF-DETR checkpoint."""

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from pmf_detr.data import IMAGE_SUFFIXES, load_image_tensor
from pmf_detr.metrics import postprocess
from pmf_detr.utils import load_model_checkpoint, select_device, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="Single 8-bit image or image directory")
    parser.add_argument("--output", default="runs/predict")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    image_size = args.imgsz if args.imgsz is not None else checkpoint["config"].get("train", {}).get("imgsz", 640)
    if image_size < 32 or image_size % 32:
        raise ValueError("--imgsz must be a positive multiple of 32.")
    names = checkpoint.get("names", [str(i) for i in range(checkpoint["config"]["model"]["num_classes"])])
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                       and output not in p.parents)
    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        paths = [source]
    else:
        raise FileNotFoundError(f"Expected a supported image or directory: {source}")
    if not paths:
        raise ValueError("Source directory contains no supported images.")
    (output / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        tensor, (width, height) = load_image_tensor(path, image_size)
        prediction = postprocess(model(tensor.unsqueeze(0).to(device)), args.max_dets, args.conf)[0]
        scale = torch.tensor([width, height, width, height], dtype=prediction["boxes"].dtype)
        pixel_boxes = prediction["boxes"] * scale
        detections = []
        with Image.open(path) as original:
            visual = original.convert("RGB")
        draw = ImageDraw.Draw(visual)
        for box, score, label in zip(pixel_boxes.tolist(), prediction["scores"].tolist(), prediction["labels"].tolist()):
            detections.append({"class_id": label, "class_name": names[label], "score": score, "xyxy": box})
            draw.rectangle(box, outline=(255, 75, 55), width=max(1, round(min(width, height) / 256)))
            draw.text((max(0, box[0]), max(0, box[1] - 12)), f"{names[label]} {score:.3f}", fill=(255, 75, 55))
        suffix = hashlib.sha256(str(path).encode()).hexdigest()[:10]
        visual_path = output / "images" / f"{path.stem}_{suffix}.png"
        visual.save(visual_path)
        records.append({"source": str(path), "width": width, "height": height,
                        "visualization": str(visual_path), "detections": detections})
    write_json({"checkpoint": str(args.checkpoint), "conf": args.conf, "imgsz": image_size,
                "box_format": "xyxy pixels in original image", "nms": False, "images": records}, output / "predictions.json")
    print(f"Saved {len(records)} image(s) and predictions to {output.resolve()}")


if __name__ == "__main__":
    main()
