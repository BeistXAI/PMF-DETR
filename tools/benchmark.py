"""Measure model-only latency with an explicit device, batch size and precision."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pmf_detr.model import build_model
from pmf_detr.utils import load_config, load_model_checkpoint, select_device, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_args = parser.add_mutually_exclusive_group()
    model_args.add_argument("--config", default=None)
    model_args.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default="runs/benchmark.json")
    args = parser.parse_args()
    if args.imgsz < 32 or args.imgsz % 32 or min(args.batch_size, args.iterations, args.threads) < 1 or args.warmup < 0:
        raise ValueError("Use a positive multiple-of-32 image size, positive batch/iterations/threads, and nonnegative warmup.")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp is supported for CUDA benchmarking only.")
    if args.checkpoint:
        model, _ = load_model_checkpoint(args.checkpoint, device)
    else:
        config_path = args.config or Path(__file__).resolve().parents[1] / "configs/pmf_detr_r18.yaml"
        model = build_model(load_config(config_path)).to(device).eval()
    images = torch.rand(args.batch_size, model.model_config["input_channels"], args.imgsz, args.imgsz, device=device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
        for _ in range(args.warmup):
            model(images)
        synchronize()
        timings = []
        for _ in range(args.iterations):
            synchronize()
            start = time.perf_counter()
            model(images)
            synchronize()
            timings.append(time.perf_counter() - start)
    mean = statistics.mean(timings)
    report = {"measurement": "model-only forward; excludes reading/resizing, postprocessing, drawing and disk writes",
              "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
              "torch": str(torch.__version__), "threads": args.threads, "precision": "CUDA autocast float16" if args.amp else "float32",
              "imgsz": args.imgsz, "batch_size": args.batch_size, "warmup": args.warmup, "iterations": args.iterations,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "mean_batch_latency_ms": mean * 1000, "median_batch_latency_ms": statistics.median(timings) * 1000,
              "throughput_images_per_second": args.batch_size / mean}
    if device.type == "cuda":
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    write_json(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
