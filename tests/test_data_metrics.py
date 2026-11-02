"""Synthetic correctness tests for annotations, AP matching, and RNG checkpointing."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pmf_detr.data import (YoloDetectionDataset, load_data_config, read_yolo_labels,
                           resolve_split, validate_split_separation)
from pmf_detr.metrics import DetectionEvaluator, postprocess
from pmf_detr.engine import _dataset_signature, evaluate, train_model
from pmf_detr.utils import (atomic_save_checkpoint, capture_rng_state, load_checkpoint,
                            load_config, restore_rng_state, seed_everything)


def prediction(boxes=(), scores=(), labels=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)}


def target(boxes=(), labels=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long)}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            Image.fromarray(np.full((20, 40), 128, dtype=np.uint8)).save(self.root / "images" / split / "slice.png")
            (self.root / "labels" / split / "slice.txt").write_text("0 0.25 0.5 0.2 0.4\n")
        self.yaml_path = self.root / "dataset.yaml"
        self.yaml_path.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [fracture]\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_grayscale_and_synchronized_flip(self):
        dataset = YoloDetectionDataset(self.yaml_path, image_size=64, augment=True, hflip=1.0)
        image, annotation = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertTrue(torch.equal(image[0], image[1]))
        self.assertAlmostEqual(float(image.mean()), 128 / 255, places=6)
        self.assertAlmostEqual(float(annotation["boxes"][0, 0]), 0.75, places=6)
        self.assertEqual(annotation["original_size"], (40, 20))

    def test_negative_is_empty_file_not_missing_file(self):
        label = self.root / "labels/train/slice.txt"
        label.write_text("")
        _, annotation = YoloDetectionDataset(self.yaml_path, image_size=64)[0]
        self.assertEqual(tuple(annotation["boxes"].shape), (0, 4))
        label.unlink()
        with self.assertRaises(FileNotFoundError):
            YoloDetectionDataset(self.yaml_path, image_size=64)

    def test_invalid_annotation_rejected(self):
        label = self.root / "labels/train/slice.txt"
        for content in ("0 0.1 0.1 0.8 0.8", "1 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 0.5 0.5 0 0.2"):
            with self.subTest(content=content):
                label.write_text(content)
                with self.assertRaises(ValueError):
                    read_yolo_labels(label, 1)

    def test_list_paths_relative_to_dataset_root(self):
        (self.root / "splits").mkdir()
        (self.root / "splits/train.txt").write_text("images/train/slice.png\n")
        config = load_data_config(self.yaml_path)
        config["train"] = "splits/train.txt"
        self.assertEqual(resolve_split(config, "train"), [(self.root / "images/train/slice.png").resolve()])

    def test_duplicate_split_leakage_rejected(self):
        config = load_data_config(self.yaml_path)
        self.assertEqual(validate_split_separation(config), {"train": 1, "val": 1})
        config["val"] = "images/train"
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            validate_split_separation(config)

    def test_high_bit_depth_requires_explicit_windowing(self):
        path = self.root / "images/train/slice.png"
        Image.fromarray(np.ones((20, 40), dtype=np.uint16) * 1024).save(path)
        with self.assertRaisesRegex(ValueError, "windowing"):
            YoloDetectionDataset(self.yaml_path, image_size=64)[0]

    def test_resume_fingerprint_detects_label_changes_at_same_path(self):
        train = YoloDetectionDataset(self.yaml_path, image_size=64)
        val = YoloDetectionDataset(self.yaml_path, "val", image_size=64)
        original = _dataset_signature(train, val)
        (self.root / "labels/train/slice.txt").write_text("0 0.5 0.5 0.2 0.4\n")
        self.assertNotEqual(original, _dataset_signature(train, val))


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        evaluator = DetectionEvaluator(1)
        evaluator.update([prediction([[.25, .25, .75, .75]], [.9], [0])],
                         [target([[.5, .5, .5, .5]], [0])])
        result = evaluator.compute()
        for key in ("map50", "map50_95", "precision", "recall", "f1"):
            self.assertEqual(result[key], 1.0)

    def test_duplicate_detection_only_matches_once(self):
        evaluator = DetectionEvaluator(1)
        evaluator.update([prediction([[.25, .25, .75, .75]] * 2, [.9, .8], [0, 0])],
                         [target([[.5, .5, .5, .5]], [0])])
        result = evaluator.compute()
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["map50"], 1.0)

    def test_confidence_order_affects_ap(self):
        evaluator = DetectionEvaluator(1)
        evaluator.update([prediction([[0, 0, .1, .1], [.25, .25, .75, .75]], [.95, .8], [0, 0])],
                         [target([[.5, .5, .5, .5]], [0])])
        self.assertEqual(evaluator.compute()["map50"], .5)

    def test_missing_detection_counts_towards_recall(self):
        evaluator = DetectionEvaluator(1)
        evaluator.update([prediction([[.25, .25, .75, .75]], [.9], [0]), prediction()],
                         [target([[.5, .5, .5, .5]], [0]), target([[.5, .5, .5, .5]], [0])])
        result = evaluator.compute()
        self.assertEqual(result["recall"], .5)
        self.assertAlmostEqual(result["map50"], 51 / 101)

    def test_ap_cutoff_independent_of_operating_threshold(self):
        evaluator = DetectionEvaluator(1, conf=.25)
        evaluator.update([prediction([[.25, .25, .75, .75]], [.2], [0])],
                         [target([[.5, .5, .5, .5]], [0])])
        result = evaluator.compute()
        self.assertEqual(result["map50"], 1.0)
        self.assertEqual(result["recall"], 0.0)

    def test_negative_slice_false_positive_and_empty_dataset(self):
        evaluator = DetectionEvaluator(1)
        evaluator.update([prediction([[.2, .2, .4, .4]], [.7], [0])], [target()])
        result = evaluator.compute()
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["map50"], 0.0)
        self.assertIsNone(result["per_class"]["0"]["ap50"])
        self.assertEqual(DetectionEvaluator(1).compute()["f1"], 0.0)

    def test_wrong_class_does_not_match(self):
        evaluator = DetectionEvaluator(2)
        evaluator.update([prediction([[.25, .25, .75, .75]], [.9], [1])],
                         [target([[.5, .5, .5, .5]], [0])])
        result = evaluator.compute()
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_postprocess_clips_boxes_and_has_no_nms(self):
        outputs = {"pred_logits": torch.tensor([[[3.], [2.]]]),
                   "pred_boxes": torch.tensor([[[0., 0., .8, .8], [0., 0., .8, .8]]])}
        predictions = postprocess(outputs, max_dets=300)[0]
        self.assertEqual(len(predictions["boxes"]), 2)
        self.assertTrue((predictions["boxes"] >= 0).all())
        self.assertEqual(float(predictions["boxes"][0, 0]), 0.0)


class CheckpointTests(unittest.TestCase):
    def test_rng_state_survives_weights_only_checkpoint(self):
        seed_everything(23)
        generator = torch.Generator().manual_seed(24)
        state = capture_rng_state(generator)
        expected = (random.random(), np.random.random(), torch.rand(4), torch.randperm(6, generator=generator))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            atomic_save_checkpoint({"format_version": 1, "model": {"weight": torch.ones(1)},
                                    "config": {"model": {}}, "rng": state}, path)
            checkpoint = load_checkpoint(path)
            restore_rng_state(checkpoint["rng"], generator)
        actual = (random.random(), np.random.random(), torch.rand(4), torch.randperm(6, generator=generator))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(torch.equal(actual[3], expected[3]))

    def test_interrupted_full_model_training_matches_uninterrupted_run(self):
        """Exercise optimizer, cosine schedule, augmentation, DN and loader RNG together."""
        from tools.make_smoke_data import make_smoke_data

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                root = Path(temporary)
                data = make_smoke_data(root / "data", seed=17)
                config = load_config(data / "smoke_config.yaml")
                config["train"]["epochs"] = 2
                uninterrupted = root / "uninterrupted"
                interrupted = root / "interrupted"
                train_model(config, data / "dataset.yaml", uninterrupted, device="cpu")

                def save_then_interrupt(checkpoint, path):
                    atomic_save_checkpoint(checkpoint, path)
                    if Path(path).name == "last.pt":
                        raise RuntimeError("simulated interruption after last.pt commit")

                with patch("pmf_detr.engine.atomic_save_checkpoint", side_effect=save_then_interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        train_model(config, data / "dataset.yaml", interrupted, device="cpu")
                self.assertTrue((interrupted / "best.pt").is_file())
                self.assertEqual(load_checkpoint(interrupted / "last.pt")["epoch"], 0)
                # Omitting output must reuse the checkpoint directory.
                train_model(config, data / "dataset.yaml", device="cpu", resume=interrupted / "last.pt")
                expected = load_checkpoint(uninterrupted / "last.pt")
                actual = load_checkpoint(interrupted / "last.pt")
                self.assertEqual(actual["epoch"], 1)
                self.assertEqual(actual["history"][-1]["lr"], expected["history"][-1]["lr"])
                self.assertEqual(actual["history"][-1]["loss"], expected["history"][-1]["loss"])
                self.assertEqual(actual["model"].keys(), expected["model"].keys())
                for name in expected["model"]:
                    self.assertTrue(torch.equal(actual["model"][name], expected["model"][name]), name)
                with self.assertRaisesRegex(ValueError, "original output directory"):
                    train_model(config, data / "dataset.yaml", root / "wrong_directory", device="cpu",
                                resume=interrupted / "last.pt")
        finally:
            torch.set_num_threads(previous_threads)


class EvaluationModeTests(unittest.TestCase):
    def test_evaluation_restores_training_flag(self):
        class FixedDetector(torch.nn.Module):
            def forward(self, images):
                self.assert_eval = not self.training
                return {"pred_logits": torch.ones(len(images), 1, 1) * 3,
                        "pred_boxes": torch.tensor([[[.5, .5, .5, .5]]]).expand(len(images), -1, -1)}

        model = FixedDetector().train()
        batches = [(torch.zeros(1, 3, 64, 64), [target([[.5, .5, .5, .5]], [0])])]
        result = evaluate(model, batches)
        self.assertTrue(model.assert_eval)
        self.assertTrue(model.training)
        self.assertEqual(result["map50"], 1.0)

    def test_evaluation_failure_also_restores_training_flag(self):
        class BrokenDetector(torch.nn.Module):
            def forward(self, images):
                raise RuntimeError("synthetic failure")

        model = BrokenDetector().train()
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            evaluate(model, [(torch.zeros(1, 3, 64, 64), [target()])])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
