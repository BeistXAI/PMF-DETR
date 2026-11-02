"""Focused mathematical/gradient checks for the three manuscript modules."""
import unittest

import torch
from torch.nn import functional as F

from pmf_detr.backbone import MEEAResNet18
from pmf_detr.model import build_model
from pmf_detr.modules import MBConv, MEEA, SADCA, SemanticDecoupling
from pmf_detr.neck import HybridEncoder


class PaperModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_meea_is_shape_preserving_gate_with_fixed_filters(self):
        module = MEEA(4)
        x = torch.randn(2, 4, 13, 17, requires_grad=True)
        y = module(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue((y.abs() <= x.abs() + 1e-7).all())
        self.assertNotIn("gabor_kernels", dict(module.named_parameters()))
        self.assertNotIn("sobel_kernels", dict(module.named_parameters()))
        y.square().mean().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertGreater(float(module.gabor_fuse.weight.grad.abs().sum()), 0)
        self.assertGreater(float(module.sobel_fuse.weight.grad.abs().sum()), 0)

    def test_dynamic_convolution_has_independent_per_image_kernels(self):
        module = SADCA(2, kernel_size=3)
        x = torch.randn(3, 2, 7, 9, requires_grad=True)
        kernels = torch.randn(3, 2 * 9, requires_grad=True)
        actual = module.dynamic_conv(x, kernels)
        expected = torch.cat([
            F.conv2d(x[i:i + 1], kernels[i].reshape(2, 1, 3, 3), padding=1, groups=2)
            for i in range(3)
        ])
        torch.testing.assert_close(actual, expected)
        # Altering another image's kernel must not change sample zero.
        changed = kernels.detach().clone()
        changed[1:] += 100
        torch.testing.assert_close(module.dynamic_conv(x, changed)[0], actual[0])
        actual.square().mean().backward()
        self.assertGreater(float(kernels.grad.abs().sum()), 0)

    def test_sadca_learns_foreground_background_and_both_attentions(self):
        x = torch.randn(2, 8, 11, 13, requires_grad=True)
        decoupler, module = SemanticDecoupling(8), SADCA(8, reduction=4)
        y = module(x, *decoupler(x))
        self.assertEqual(x.shape, y.shape)
        y.square().mean().backward()
        for prefix, item in (("decoupler", decoupler), ("SADCA", module)):
            for name, parameter in item.named_parameters():
                self.assertIsNotNone(parameter.grad, f"Missing gradient: {prefix}.{name}")
                self.assertTrue(torch.isfinite(parameter.grad).all(), f"Bad gradient: {prefix}.{name}")
        self.assertGreater(float(module.fg_kernel.weight.grad.abs().sum()), 0)
        self.assertGreater(float(module.bg_kernel.weight.grad.abs().sum()), 0)

    def test_mbconv_diagonal_bias_is_initialization_not_a_mask(self):
        torch.manual_seed(21)
        plain = MBConv(8, 12, diagonal_scale=1)
        torch.manual_seed(21)
        biased = MBConv(8, 12, diagonal_scale=2)
        for name, mask in (("diagonal", torch.eye(3)), ("antidiagonal", torch.eye(3).flip(1))):
            p, b = getattr(plain, name)[0].weight, getattr(biased, name)[0].weight
            torch.testing.assert_close(b, p * (1 + mask))
        x = torch.randn(2, 8, 11, 15, requires_grad=True)
        y = biased(x)
        self.assertEqual(tuple(y.shape), (2, 12, 11, 15))
        y.square().mean().backward()
        off_diagonal = 1 - torch.eye(3)
        gradient = biased.diagonal[0].weight.grad
        self.assertGreater(float((gradient * off_diagonal).abs().sum()), 0)

    def test_four_backbone_levels_and_p2_contribution_to_decoder_levels(self):
        backbone = MEEAResNet18().eval()
        with torch.no_grad():
            features = backbone(torch.randn(1, 3, 64, 96))
        self.assertEqual([tuple(x.shape) for x in features],
                         [(1, 64, 16, 24), (1, 128, 8, 12),
                          (1, 256, 4, 6), (1, 512, 2, 3)])
        neck = HybridEncoder(hidden_dim=16, num_heads=4, dim_feedforward=32, fpn_depth=1).eval()
        features = [f.detach().requires_grad_() for f in features]
        outputs = neck(features)
        self.assertEqual([tuple(x.shape) for x in outputs],
                         [(1, 16, 8, 12), (1, 16, 4, 6), (1, 16, 2, 3)])
        sum(x.square().mean() for x in outputs).backward()
        for index, feature in enumerate(features):
            self.assertGreater(float(feature.grad.abs().sum()), 0, f"Unused P{index + 2}")

    def test_model_config_round_trip_and_forward(self):
        config = {"model": {"hidden_dim": 32, "num_heads": 4, "num_queries": 10,
                            "num_decoder_layers": 1, "dim_feedforward": 64,
                            "num_denoising": 0, "fpn_depth": 1}}
        model = build_model(config).eval()
        replica = build_model(model.model_config).eval()
        replica.load_state_dict(model.state_dict(), strict=True)
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected, actual = model(image), replica(image)
        self.assertEqual(tuple(actual["pred_logits"].shape), (1, 10, 1))
        self.assertEqual(tuple(actual["pred_boxes"].shape), (1, 10, 4))
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(actual[key], expected[key])
        self.assertTrue(((actual["pred_boxes"] >= 0) & (actual["pred_boxes"] <= 1)).all())
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model({"model": {"hiddden_dim": 64}})


if __name__ == "__main__":
    unittest.main()
