from pathlib import Path

from tools.benchmark_five_method_latency import (
    METHOD_ORDER,
    STAGE_ORDER,
    build_aggregate_command,
    build_run_commands,
    build_summary_command,
    completed_run,
    resolve_executable_path,
)


def test_python_executable_path_does_not_dereference_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-target"
    target.touch()
    link = tmp_path / "venv-python"
    link.symlink_to(target)

    assert resolve_executable_path(str(link)) == link


def test_completed_run_requires_full_manifest_coverage(tmp_path: Path) -> None:
    output = tmp_path / "method"
    output.mkdir()
    assert not completed_run(output, expected_queries=128)

    queries = output / "queries"
    queries.mkdir()
    (output / "run_manifest.json").write_text(
        '{"identity": {"query_indices": [1, 2]}}\n'
    )
    (queries / "query_00001.json").touch()
    assert not completed_run(output, expected_queries=2)

    (queries / "query_00002.json").touch()
    assert completed_run(output, expected_queries=2)


def test_unified_commands_cover_five_methods_and_one_summary(tmp_path: Path) -> None:
    config = {
        "device": "cuda:0",
        "cuda_visible_devices": "2",
        "candidate_batch_size": 64,
        "score_seed": 42,
        "tau": 0.1,
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
        "fem_primary_dir": shared / "fem-primary",
        "fem_oversized_dir": shared / "fem-oversized",
        "fem_external_runtime": shared / "fem-external.json",
        "fem_fallback_runtime": shared / "fem-fallback.json",
        "selector_latency": shared / "selector.json",
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
        "fem_omp",
    )
    assert tuple(command.name for command in commands) == STAGE_ORDER
    assert len(commands) == 4
    for command in commands[:4]:
        assert "--pilot-manifest" in command.command
        assert str(paths["selection"]) in command.command
        assert command.environment["CUDA_VISIBLE_DEVICES"] == "2"
    for command in commands[:3]:
        assert "--score-sample-counts" in command.command
        assert command.command[command.command.index("--score-sample-counts") + 1] == "1"
        assert "--synchronize-latency" in command.command
        assert "--measure-core-forward" in command.command
    assert "--context-counts" in commands[3].command
    assert commands[3].command[commands[3].command.index("--context-counts") + 1] == "8"
    assert "--measure-core-forward" in commands[3].command

    summary = build_summary_command(paths, selection=paths["selection"], output_dir=output)
    assert summary.name == "summarize"
    assert str(output / "summary.json") in summary.command
    assert str(paths["fem_primary_dir"]) in summary.command
    assert str(paths["fem_oversized_dir"]) in summary.command
    assert str(paths["fem_external_runtime"]) in summary.command
    assert "--fem-fallback-runtime" in summary.command
    assert str(paths["fem_fallback_runtime"]) in summary.command
    assert "--selector-latency" in summary.command
    assert str(paths["selector_latency"]) in summary.command

    aggregate = build_aggregate_command(
        paths,
        summaries=[output / "summary.json", tmp_path / "repeat_002" / "summary.json"],
        output_root=tmp_path,
    )
    assert aggregate.name == "aggregate"
    assert aggregate.command.count("--summary") == 2
    assert str(tmp_path / "summary_final.json") in aggregate.command
