#!/usr/bin/env python3
"""Evaluate a saved PMF-DETR checkpoint without changing the training split."""

import argparse
import json
import torch

from pmf_detr.data import YoloDetectionDataset, load_data_config, validate_split_separation
from pmf_detr.engine import EVAL_DEFAULTS, evaluate, make_loader
from pmf_detr.utils import load_model_checkpoint, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float, help="P/R/F1 operating threshold, independent of AP")
    parser.add_argument("--score-threshold", type=float, help="Low score cutoff for AP; default from checkpoint or 0.001")
    parser.add_argument("--max-dets", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", default="runs/val/metrics.json")
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    data = load_data_config(args.data)
    validate_split_separation(data)
    if data["names"] != checkpoint.get("names"):
        raise ValueError("Dataset class names/order differ from the checkpoint.")
    image_size = args.imgsz if args.imgsz is not None else checkpoint["config"].get("train", {}).get("imgsz", 640)
    dataset = YoloDetectionDataset(data, args.split, image_size)
    loader = make_loader(dataset, args.batch_size, args.workers, device=device)
    settings = {**EVAL_DEFAULTS, **checkpoint["config"].get("eval", {})}
    for key in ("conf", "score_threshold", "max_dets"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    metrics = evaluate(model, loader, device, len(data["names"]), data["names"], **settings)
    metrics.update(checkpoint=str(args.checkpoint), split=args.split, imgsz=image_size,
                   metric_definition="101-point all-size AP; no crowd/ignore or COCO area stratification")
    write_json(metrics, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
