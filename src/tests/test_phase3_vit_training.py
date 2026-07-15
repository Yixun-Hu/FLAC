import importlib.util
import math
import types
import unittest
from pathlib import Path

from torch import nn


SCRIPT = Path(__file__).resolve().parents[2] / "worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py"
SPEC = importlib.util.spec_from_file_location("train_vit_ablation", SCRIPT)
train_vit_ablation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_vit_ablation)


class DummyDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        shared_geometry = nn.Linear(3, 4)
        self.model = nn.Linear(4, 4)
        self.conditioner = types.SimpleNamespace(
            conditioners={
                "source_vit": shared_geometry,
                "context_poses_vit": shared_geometry,
                "context_audio": nn.Linear(4, 4),
            }
        )
        self.conditioner.parameters = lambda: (
            param
            for module in self.conditioner.conditioners.values()
            for param in module.parameters()
        )

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        yield from self.model.named_parameters(prefix="model")
        seen = set()
        for key, module in self.conditioner.conditioners.items():
            for name, param in module.named_parameters(prefix=f"conditioner.{key}"):
                if id(param) not in seen:
                    seen.add(id(param))
                    yield name, param


def _dummy_module():
    return types.SimpleNamespace(diffusion=DummyDiffusion())


def _optimizer_args():
    return types.SimpleNamespace(
        geometry_lr=2e-5,
        dit_lr=2e-6,
        other_cond_lr=1e-6,
        lr=5e-6,
        warmup_steps=500,
        max_steps=15_000,
        min_lr_ratio=0.1,
        scheduler="cosine",
        smoke=False,
    )


class Phase3TrainingTest(unittest.TestCase):
    def test_phase_lr_multiplier_warmup_and_cosine_boundaries(self):
        fn = train_vit_ablation.phase_lr_multiplier
        self.assertTrue(math.isclose(fn(0, 500, 15_000, 0.1, "cosine"), 1 / 500))
        self.assertTrue(math.isclose(fn(499, 500, 15_000, 0.1, "cosine"), 1.0))
        self.assertTrue(math.isclose(fn(500, 500, 15_000, 0.1, "cosine"), 1.0))
        self.assertTrue(math.isclose(fn(15_000, 500, 15_000, 0.1, "cosine"), 0.1))

    def test_parameter_groups_are_disjoint_and_deduplicate_shared_geometry(self):
        module = _dummy_module()
        groups, summary = train_vit_ablation.build_parameter_groups(module, 2e-5, 2e-6, 1e-6)
        ids = [id(param) for group in groups for param in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            [group["name"] for group in groups],
            ["geometry", "dit", "other_conditioners"],
        )
        self.assertEqual([group["lr"] for group in groups], [2e-5, 2e-6, 1e-6])
        self.assertEqual(
            summary["geometry"]["parameters"],
            sum(p.numel() for p in module.diffusion.conditioner.conditioners["source_vit"].parameters()),
        )

    def test_installed_optimizer_preserves_group_lr_ratios_through_warmup(self):
        module = _dummy_module()
        model_config = {
            "training": {
                "optimizer_configs": {
                    "diffusion": {
                        "optimizer": {
                            "config": {"betas": [0.9, 0.999], "weight_decay": 1e-3}
                        }
                    }
                }
            }
        }
        train_vit_ablation.install_phase3_optimizer(module, _optimizer_args(), model_config)
        configured = module.configure_optimizers()
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        expected = [2e-5 / 500, 2e-6 / 500, 1e-6 / 500]
        self.assertEqual([group["name"] for group in optimizer.param_groups], [
            "geometry", "dit", "other_conditioners"
        ])
        for actual, target in zip([group["lr"] for group in optimizer.param_groups], expected):
            self.assertTrue(math.isclose(actual, target))
        optimizer.step()
        scheduler.step()
        for actual, target in zip(
            [group["lr"] for group in optimizer.param_groups],
            [value * 2 for value in expected],
        ):
            self.assertTrue(math.isclose(actual, target))


if __name__ == "__main__":
    unittest.main()
