from pathlib import Path

from tools.benchmark_five_method_latency import (
    METHOD_ORDER,
    STAGE_ORDER,
    build_run_commands,
    build_summary_command,
    canonical_sha256,
    make_warmup_selection,
    resolve_executable_path,
)


def test_make_warmup_selection_rehashes_prefix() -> None:
    source = {
        "query_count": 2,
        "room_count": 2,
        "candidate_query_pairs": 30,
        "records": [
            {"index": 1, "room": "room_a", "candidate_count": 10},
            {"index": 2, "room": "room_b", "candidate_count": 20},
        ],
    }
    source["sha256"] = canonical_sha256(source)

    warmup = make_warmup_selection(source, 1)

    assert warmup["query_count"] == 1
    assert warmup["room_count"] == 1
    assert warmup["candidate_query_pairs"] == 10
    assert warmup["parent_selection_sha256"] == source["sha256"]
    body = {key: value for key, value in warmup.items() if key != "sha256"}
    assert warmup["sha256"] == canonical_sha256(body)


def test_python_executable_path_does_not_dereference_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-target"
    target.touch()
    link = tmp_path / "venv-python"
    link.symlink_to(target)

    assert resolve_executable_path(str(link)) == link


def test_unified_commands_cover_five_methods_and_one_summary(tmp_path: Path) -> None:
    config = {
        "device": "cuda:0",
        "cuda_visible_devices": "2",
        "candidate_batch_size": 64,
        "agree_candidate_batch_size": 32,
        "score_seed": 42,
        "tau": 0.1,
        "fem_workers": 1,
        "fem_solver_threads": 24,
        "methods": {
            "vanilla_flac": {"conditioning_method": "vanilla"},
            "fa_bf_flac": {"conditioning_method": "fa_invariant"},
            "yawaug_flac": {"conditioning_method": "vanilla"},
            "few_shot_rir": {},
        },
    }
    shared = tmp_path / "shared"
    paths = {
        "python": Path("/python"),
        "selection": shared / "selection.json",
        "context_manifest": shared / "contexts.json",
        "geometry_audit": shared / "geometry.json",
        "dataset_root": shared / "dataset",
        "agree_checkpoint": shared / "agree.pt",
        "mkl_runtime": shared / "libmkl.so",
        "vanilla_flac.model_config": shared / "flac.json",
        "vanilla_flac.checkpoint": shared / "vanilla.ckpt",
        "fa_bf_flac.model_config": shared / "flac.json",
        "fa_bf_flac.checkpoint": shared / "fa.ckpt",
        "yawaug_flac.model_config": shared / "flac.json",
        "yawaug_flac.checkpoint": shared / "yaw.ckpt",
        "few_shot_rir.model_config": shared / "few.json",
        "few_shot_rir.checkpoint": shared / "few.ckpt",
    }
    output = tmp_path / "repeat_001"

    commands = build_run_commands(
        config, paths, selection=paths["selection"], output_dir=output
    )

    assert METHOD_ORDER == (
        "vanilla_flac",
        "fa_bf_flac",
        "yawaug_flac",
        "few_shot_rir",
        "fem_agree",
    )
    assert tuple(command.name for command in commands) == STAGE_ORDER
    assert len(commands) == 6
    for command in commands[:4]:
        assert "--pilot-manifest" in command.command
        assert str(paths["selection"]) in command.command
        assert command.environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert "--workers" in commands[4].command
    assert "--stage" in commands[5].command
    assert "all" in commands[5].command

    summary = build_summary_command(paths, selection=paths["selection"], output_dir=output)
    assert summary.name == "summarize"
    assert str(output / "summary.json") in summary.command
    assert str(output / "fem_agree" / "results") in summary.command
