"""exp_46 G7 -- the registered loader lengths are the production loader's (plan_c4flip_data_curve.md r3 G7, §5 L0:
"confirms G7's count by a CPU job").

For every registered fraction (025 / 050 / 075 / 100) the PRODUCTION path builds the training DataLoader exactly as
train.py does (``create_dataloader_from_config`` with the run's dataset config, the C4F model config's sample size /
rate / channels, micro-batch 32, 6 workers), and PL 2.1's own sampler injection re-instantiates it per rank for the
registered 2 ranks (``_get_distributed_sampler`` + ``_update_dataloader``, as the DDP strategy does at fit, mode
TRAINING, shuffle on, seed 42): ``len(loader)`` on each rank must be ``FRACS[tag]["steps_per_epoch"]`` -- 1,148 /
2,276 / 3,413 / 4,550 -- = floor(ceil(N / 2) / 32), and the 40,000-step checkpoint is epoch=34 / 17 / 11 / 8.

Light on I/O by construction and asserted: the dataset constructor reads the split JSON and imports the metadata module
(json_scandir only builds path strings); an audit hook proves no file under AcousticRooms is opened (no waveform,
depth or metadata read, no fan-out on /n/fs).
"""
import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402

FINAL = {"025": "epoch=34-step=40000.ckpt", "050": "epoch=17-step=40000.ckpt", "075": "epoch=11-step=40000.ckpt",
         "100": "epoch=8-step=40000.ckpt"}
STEPS = {"025": 1148, "050": 2276, "075": 3413, "100": 4550}
_OPENED = {"active": False, "paths": []}


def _hook(event, args):
    if _OPENED["active"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED["paths"].append(os.fsdecode(args[0]))


sys.addaudithook(_hook)


def test_the_registered_fractions_are_the_four_c4f_points():
    assert sorted(reg.FRACS) == sorted(STEPS) and {t: f["steps_per_epoch"] for t, f in reg.FRACS.items()} == STEPS
    assert reg.FRACS["100"]["n_items"] == 291210 and math.ceil(291210 / 2) // 32 == 4550


@pytest.mark.parametrize("tag", sorted(STEPS))
def test_the_production_loader_on_each_of_the_two_ranks_has_the_registered_length(tag, monkeypatch):
    from pytorch_lightning.trainer.connectors.data_connector import _get_distributed_sampler
    from pytorch_lightning.trainer.states import RunningStage
    from pytorch_lightning.utilities.data import _update_dataloader
    from src.data.dataset import create_dataloader_from_config
    f = reg.FRACS[tag]
    model = json.load(open(os.path.join(REPO, reg.ARMS["c4f"]["config"])))
    dataset_config = json.load(open(os.path.join(REPO, f["dataset_config"])))
    assert reg.sha256_file(os.path.join(REPO, f["dataset_config"])) == f["dataset_config_sha256"]
    monkeypatch.chdir(REPO)                                     # the configs' paths are cwd-relative, as in a run
    monkeypatch.setenv("PL_GLOBAL_SEED", str(reg.SEED))
    _OPENED["paths"].clear()
    _OPENED["active"] = True
    try:
        dl = create_dataloader_from_config(dataset_config, batch_size=32, num_workers=6,
                                           sample_rate=model["sample_rate"], sample_size=model["sample_size"],
                                           audio_channels=model.get("audio_channels", 1))
        n = len(dl.dataset)
        lengths = []
        for rank in (0, 1):
            sampler = _get_distributed_sampler(dl, True, mode=RunningStage.TRAINING, overfit_batches=0,
                                               num_replicas=2, rank=rank)
            lengths.append(len(_update_dataloader(dl, sampler, mode=RunningStage.TRAINING)))
    finally:
        _OPENED["active"] = False
    assert n == f["n_items"] and dl.drop_last is True and dl.batch_size == 32
    assert lengths == [STEPS[tag]] * 2 == [math.ceil(n / 2) // 32] * 2, (tag, n, lengths)
    assert reg.ckpt_name(reg.MAX_STEPS, tag) == FINAL[tag] and reg.epoch_of_step(reg.MAX_STEPS, tag) == int(
        FINAL[tag].split("=")[1].split("-")[0])
    touched = sorted({p for p in _OPENED["paths"] if "AcousticRooms" in p})
    assert touched == [], touched[:5]                           # no dataset file was opened
    assert os.path.realpath(os.path.join(REPO, f["split"])) in {os.path.realpath(p) for p in _OPENED["paths"]}
