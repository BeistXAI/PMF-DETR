"""Geometry, assignment, denoising and differentiable decoder regression tests."""

import math
import unittest

import torch

from pmf_detr.box_ops import (aligned_iou_and_giou, box_cxcywh_to_xyxy,
                              focaler_giou_loss, normalized_wasserstein_loss)
from pmf_detr.criterion import HungarianMatcher, SetCriterion, build_criterion
from pmf_detr.decoder import RTDETRDecoder, build_denoising_group


def targets_mixed():
    return [
        {"labels": torch.tensor([0, 1]), "boxes": torch.tensor([[.3, .4, .2, .1], [.7, .6, .1, .2]])},
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
    ]


def require_cpu_half_conversion(test_case):
    """Some sandboxed CPUs hide /proc/cpuinfo and PyTorch cannot convert fp16."""
    try:
        torch.ones(1).half().float()
    except RuntimeError as exc:
        if "Failed to initialize cpuinfo" in str(exc):
            test_case.skipTest("PyTorch fp16 CPU conversion unavailable: cpuinfo initialization failed.")
        raise


class GeometryTests(unittest.TestCase):
    def test_identical_boxes_have_zero_focaler_loss(self):
        boxes = torch.tensor([[.5, .5, .2, .3], [.2, .2, .1, .1], [.5, .5, .0001, .0001]])
        torch.testing.assert_close(focaler_giou_loss(boxes, boxes), torch.zeros(3), atol=1e-6, rtol=0)

    def test_disjoint_boxes_retain_localization_gradient(self):
        predicted = torch.tensor([[.2, .2, .1, .1]], requires_grad=True)
        target = torch.tensor([[.8, .8, .1, .1]])
        loss = focaler_giou_loss(predicted, target).sum()
        self.assertGreater(float(loss), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(float(predicted.grad.abs().sum()), 0)

    def test_full_interval_matches_giou(self):
        predicted = torch.tensor([[.5, .5, .3, .4], [.1, .1, .1, .1]])
        target = torch.tensor([[.55, .52, .2, .2], [.8, .8, .1, .1]])
        _, giou = aligned_iou_and_giou(box_cxcywh_to_xyxy(predicted), box_cxcywh_to_xyxy(target))
        torch.testing.assert_close(focaler_giou_loss(predicted, target, 0, 1), 1 - giou)

    def test_invalid_focaler_interval_rejected(self):
        box = torch.tensor([[.5, .5, .2, .2]])
        for lower, upper in ((.5, .5), (.6, .5), (-.1, .9), (0, 1.1)):
            with self.assertRaises(ValueError):
                focaler_giou_loss(box, box, lower, upper)

    def test_nwd_constant_is_in_pixels(self):
        predicted = torch.tensor([[.4, .5, .2, .2]])
        target = torch.tensor([[.5, .5, .2, .2]])
        expected = 1 - math.exp(-64 / 12.8)
        self.assertAlmostEqual(float(normalized_wasserstein_loss(predicted, target, 640, 12.8)), expected, places=6)
        self.assertEqual(float(normalized_wasserstein_loss(target, target)), 0.0)


class MatcherAndLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matcher_selects_correct_box_and_handles_empty_image(self):
        outputs = {"pred_logits": torch.tensor([[[0.], [0.]], [[0.], [0.]]]),
                   "pred_boxes": torch.tensor([[[.1, .1, .1, .1], [.5, .5, .2, .2]],
                                                [[.1, .1, .1, .1], [.5, .5, .2, .2]]])}
        targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[.5, .5, .2, .2]])},
                   {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
        assignments = HungarianMatcher()(outputs, targets)
        self.assertEqual(assignments[0][0].tolist(), [1])
        self.assertEqual(assignments[0][1].tolist(), [0])
        self.assertEqual(assignments[1][0].numel(), 0)

    def test_empty_targets_still_train_background(self):
        logits = torch.zeros(2, 5, 1, requires_grad=True)
        boxes = torch.full((2, 5, 4), .3, requires_grad=True)
        outputs = {"pred_logits": logits, "pred_boxes": boxes}
        targets = [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)} for _ in range(2)]
        loss = SetCriterion(1)(outputs, targets)
        self.assertGreater(float(loss["loss_class"]), 0)
        self.assertEqual(float(loss["loss_bbox"]), 0)
        loss["loss_total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(boxes.grad).all())

    def test_invalid_targets_and_nonfinite_predictions_rejected(self):
        outputs = {"pred_logits": torch.zeros(1, 2, 1), "pred_boxes": torch.full((1, 2, 4), .3)}
        with self.assertRaises(ValueError):
            SetCriterion(1)(outputs, [{"labels": torch.tensor([1]), "boxes": torch.tensor([[.5, .5, .2, .2]])}])
        outputs["pred_logits"][0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            SetCriterion(1)(outputs, [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}])

    def test_focal_baseline_positive_targets_is_defined(self):
        outputs = {"pred_logits": torch.zeros(1, 2, 1, requires_grad=True),
                   "pred_boxes": torch.tensor([[[.5, .5, .2, .2], [.1, .1, .1, .1]]], requires_grad=True)}
        targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[.5, .5, .2, .2]])}]
        loss = SetCriterion(1, class_loss="focal", box_loss="giou")(outputs, targets)
        self.assertTrue(torch.isfinite(loss["loss_total"]))
        loss["loss_total"].backward()

    def test_config_rejects_unknown_loss_setting(self):
        with self.assertRaisesRegex(ValueError, "Unknown loss"):
            build_criterion({"loss": {"focaler_lowre": 0.1}}, 1)
        criterion = build_criterion({"loss": {"box_loss": "giou"}, "train": {"imgsz": 320}}, 1)
        self.assertEqual(criterion.image_size, 320)

    def test_half_prediction_loss_accumulates_finite_float32(self):
        # Tests loss precision promotion only; a CPU test cannot validate CUDA AMP.
        require_cpu_half_conversion(self)
        logits = torch.zeros(2, 3, 2, dtype=torch.float16, requires_grad=True)
        boxes = torch.tensor([[[.3, .4, .2, .1], [.7, .6, .1, .2], [.1, .1, .1, .1]]],
                             dtype=torch.float16).repeat(2, 1, 1).requires_grad_()
        losses = SetCriterion(2)({"pred_logits": logits, "pred_boxes": boxes}, targets_mixed())
        self.assertEqual(losses["loss_total"].dtype, torch.float32)
        losses["loss_total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(boxes.grad).all())


class DecoderAndDenoisingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rectangular_anchor_coordinates(self):
        logits, valid = RTDETRDecoder.generate_anchors([(2, 4)], torch.device("cpu"), torch.float32)
        anchors = logits.sigmoid()
        torch.testing.assert_close(anchors[0, 0, :2], torch.tensor([.125, .25]))
        torch.testing.assert_close(anchors[0, -1, :2], torch.tensor([.875, .75]))
        self.assertTrue(valid.all())

    def test_known_dn_assignments_and_group_isolation(self):
        embedding = torch.nn.Embedding(3, 16)
        _, _, mask, metadata = build_denoising_group(targets_mixed(), embedding, 10, 4, 0, 0)
        self.assertEqual(metadata["num_groups"], 2)
        self.assertEqual(metadata["positive_indices"][0].tolist(), [0, 1, 4, 5])
        self.assertEqual(metadata["target_indices"][0].tolist(), [0, 1, 0, 1])
        self.assertEqual(metadata["positive_indices"][1].numel(), 0)
        self.assertFalse(bool(metadata["valid_mask"][1].any()))
        self.assertFalse(bool(mask[0, 2]))  # own negative group
        self.assertTrue(bool(mask[0, 4]))  # another group's positive queries
        self.assertTrue(bool(mask[8:, :8].all()))  # matching cannot see GT-derived queries
        self.assertFalse(bool(mask[:8, 8:].any()))

    def test_full_backward_auxiliary_dn_and_padding(self):
        torch.manual_seed(7)
        decoder = RTDETRDecoder(2, hidden_dim=32, num_queries=10, num_decoder_layers=2,
                                num_heads=4, num_points=2, dim_feedforward=64, num_denoising=4)
        features = [torch.randn(2, 32, size, size, requires_grad=True) for size in (8, 4, 2)]
        targets = targets_mixed()
        outputs = decoder(features, targets)

        class CountingMatcher(HungarianMatcher):
            calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                return super().forward(*args, **kwargs)

        matcher = CountingMatcher()
        criterion = SetCriterion(2, matcher=matcher)
        losses = criterion(outputs, targets)
        self.assertEqual(matcher.calls, 3)  # last decoder + one auxiliary + encoder; DN is known
        self.assertIn("loss_class_dn_0", losses)
        self.assertIn("loss_giou_dn_1", losses)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        torch.testing.assert_close(losses["loss_total"], sum(value for key, value in losses.items() if key != "loss_total"))
        losses["loss_total"].backward()
        self.assertTrue(all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features))

        # Empty-image DN padding must not become background training examples.
        changed = dict(outputs)
        changed["dn_outputs"] = [{key: value.detach().clone() for key, value in output.items()}
                                 for output in outputs["dn_outputs"]]
        for output in changed["dn_outputs"]:
            output["pred_logits"][1] = 50
        changed_losses = criterion(changed, targets)
        for layer in (0, 1):
            torch.testing.assert_close(changed_losses[f"loss_class_dn_{layer}"], losses[f"loss_class_dn_{layer}"])
        decoder.eval()
        with torch.no_grad():
            prediction = decoder([feature.detach() for feature in features])
        self.assertEqual(prediction["pred_boxes"].shape, (2, 10, 4))
        self.assertEqual(prediction["aux_outputs"], [])
        self.assertEqual(prediction["dn_outputs"], [])

    def test_small_feature_pyramid_error_is_actionable(self):
        decoder = RTDETRDecoder(1, hidden_dim=32, num_queries=300, num_heads=4,
                                num_decoder_layers=1, dim_feedforward=64)
        with self.assertRaisesRegex(ValueError, "fewer than num_queries"):
            decoder([torch.randn(2, 32, size, size) for size in (8, 4, 2)])

    def test_last_layer_box_loss_reaches_previous_box_head(self):
        decoder = RTDETRDecoder(1, hidden_dim=16, num_queries=5, num_heads=4,
                                num_points=2, num_decoder_layers=2, dim_feedforward=32,
                                num_denoising=0)
        features = [torch.randn(2, 16, size, size) for size in (4, 2, 1)]
        outputs = decoder(features)
        outputs["pred_boxes"].sum().backward()
        gradient = decoder.dec_bbox_head[0].layers[-1].bias.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_half_denoising_metadata_and_boxes_are_finite(self):
        require_cpu_half_conversion(self)
        embedding = torch.nn.Embedding(3, 16).half()
        embeddings, references, mask, metadata = build_denoising_group(targets_mixed(), embedding, 10, 4)
        self.assertEqual(embeddings.dtype, torch.float16)
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(torch.isfinite(references).all())
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(metadata["positive_indices"][0].device, embeddings.device)


if __name__ == "__main__":
    unittest.main()
