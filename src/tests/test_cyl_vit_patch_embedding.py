import copy
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from src.models.cyl_vit import CylindricalCNNPatchEmbedding, CylindricalViT
from src.models.conditioners import create_multi_conditioner_from_conditioning_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/model_configs/FLAC/AR"


class CylindricalPatchEmbeddingTest(unittest.TestCase):
    def test_linear_remains_default_and_is_parameter_matched_to_cnn(self):
        linear = CylindricalViT(depth=0)
        cnn = CylindricalViT(depth=0, patch_embed_type="cnn")

        self.assertEqual(linear.patch_embed_type, "linear")
        self.assertEqual(cnn.patch_embed_type, "cnn")
        self.assertEqual(sum(p.numel() for p in linear.to_patch_embedding.parameters()), 791_040)
        self.assertEqual(sum(p.numel() for p in cnn.to_patch_embedding.parameters()), 791_050)

    def test_cnn_stem_contract_and_token_shape(self):
        model = CylindricalViT(
            image_size=(16, 512),
            depth=0,
            patch_embed_type="cnn",
        )
        stem = model.to_patch_embedding
        self.assertIsInstance(stem, CylindricalCNNPatchEmbedding)

        expected = [
            (3, 34, (3, 3), (2, 2), 1, 1, nn.GELU),
            (34, 44, (3, 3), (2, 2), 1, 1, nn.GELU),
            (44, 96, (3, 3), (2, 2), 1, 1, nn.GELU),
            (96, 512, (3, 5), (2, 4), 1, 2, nn.Identity),
        ]
        for stage, contract in zip(stem.stages, expected):
            in_ch, out_ch, kernel, stride, pad_h, pad_w, activation = contract
            self.assertEqual(stage.conv.in_channels, in_ch)
            self.assertEqual(stage.conv.out_channels, out_ch)
            self.assertEqual(stage.conv.kernel_size, kernel)
            self.assertEqual(stage.conv.stride, stride)
            self.assertEqual(stage.conv.padding, (0, 0))
            self.assertIsNone(stage.conv.bias)
            self.assertEqual(stage.height_padding, pad_h)
            self.assertEqual(stage.width_padding, pad_w)
            self.assertIsInstance(stage.activation, activation)

        with torch.no_grad():
            tokens = stem(torch.randn(1, 3, 16, 512))
        self.assertEqual(tokens.shape, (1, 16, 512))

    def test_cnn_variant_is_c16_equivariant_after_gauge_alignment(self):
        torch.manual_seed(42)
        model = CylindricalViT(
            image_size=(16, 512),
            depth=0,
            patch_embed_type="cnn",
        ).eval()
        geometry = torch.randn(1, 3, 16, 512)

        c16_orbit = []
        for token_shift in range(16):
            pixel_shift = 32 * token_shift
            alpha = 2.0 * math.pi * pixel_shift / geometry.shape[-1]
            yaw_rotated = torch.roll(geometry, shifts=pixel_shift, dims=-1).clone()
            rolled_x = yaw_rotated[:, 0].clone()
            rolled_y = yaw_rotated[:, 1].clone()
            yaw_rotated[:, 0] = math.cos(alpha) * rolled_x - math.sin(alpha) * rolled_y
            yaw_rotated[:, 1] = math.sin(alpha) * rolled_x + math.cos(alpha) * rolled_y
            c16_orbit.append(yaw_rotated)

        with torch.no_grad():
            outputs = model(torch.cat(c16_orbit, dim=0)).reshape(16, 1, 16, 512)

        base = outputs[:1]
        expected = torch.cat(
            [torch.roll(base, shifts=token_shift, dims=2) for token_shift in range(16)],
            dim=0,
        )
        torch.testing.assert_close(outputs, expected, rtol=1e-5, atol=3e-5)

        pooled = outputs.reshape(16, 16, 512).mean(dim=1)
        torch.testing.assert_close(
            pooled,
            pooled[:1].expand_as(pooled),
            rtol=1e-5,
            atol=3e-5,
        )

    def test_experiment_configs_only_change_declared_vit_initialization_fields(self):
        with open(CONFIG_DIR / "FLAC_AR_CylViT.json") as handle:
            baseline = json.load(handle)

        for filename, patch_embed_type in (
            ("FLAC_AR_CylViT_PE_Linear.json", "linear"),
            ("FLAC_AR_CylViT_PE_CNN.json", "cnn"),
        ):
            with self.subTest(filename=filename):
                with open(CONFIG_DIR / filename) as handle:
                    variant = json.load(handle)
                vit_configs = [
                    item["config"]["ViT"]
                    for item in variant["model"]["conditioning"]["configs"]
                    if item["type"] == "ViTCoordinates"
                ]
                self.assertEqual(len(vit_configs), 2)
                self.assertTrue(all(vit["from_scratch"] for vit in vit_configs))
                self.assertTrue(
                    all(vit["patch_embed_type"] == patch_embed_type for vit in vit_configs)
                )

                normalized = copy.deepcopy(variant)
                for item in normalized["model"]["conditioning"]["configs"]:
                    if item["type"] == "ViTCoordinates":
                        vit = item["config"]["ViT"]
                        vit["from_scratch"] = False
                        vit.pop("patch_embed_type")
                self.assertEqual(normalized, baseline)

    def test_conditioner_factory_forwards_patch_embed_type(self):
        conditioning_config = {
            "cond_dim": 256,
            "configs": [
                {
                    "id": "source_vit",
                    "type": "ViTCoordinates",
                    "config": {
                        "ViT": {
                            "arch": "cyl_vit",
                            "ch_dim": 3,
                            "img_h": 256,
                            "img_w": 512,
                            "patch_h": 16,
                            "patch_w": 32,
                            "dim": 512,
                            "depth": 12,
                            "heads": 8,
                            "mlp_dim": 512,
                            "patch_embed_type": "cnn",
                        },
                        "max_value": 1,
                        "token_pool": "mean",
                    },
                }
            ],
        }
        fake_vit = nn.Module()
        fake_vit.num_tokens = 256
        with mock.patch("src.models.conditioners.CylindricalViT", return_value=fake_vit) as ctor:
            create_multi_conditioner_from_conditioning_config(conditioning_config)
        self.assertEqual(ctor.call_args.kwargs["patch_embed_type"], "cnn")


if __name__ == "__main__":
    unittest.main()
