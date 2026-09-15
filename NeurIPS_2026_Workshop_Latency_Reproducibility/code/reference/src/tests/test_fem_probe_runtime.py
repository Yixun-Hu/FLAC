import pytest

import probe_fem_rooms


def test_parallel_resource_gate_rejects_cpu_oversubscription(monkeypatch):
    monkeypatch.setattr(probe_fem_rooms.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(probe_fem_rooms, "_available_memory_gib", lambda: 256.0)

    with pytest.raises(ValueError, match="logical CPUs"):
        probe_fem_rooms._validate_parallel_resources(
            room_workers=2,
            solver_threads=8,
            minimum_memory_gib_per_worker=32.0,
        )


def test_parallel_resource_gate_rejects_insufficient_available_memory(monkeypatch):
    monkeypatch.setattr(probe_fem_rooms.os, "cpu_count", lambda: 48)
    monkeypatch.setattr(probe_fem_rooms, "_available_memory_gib", lambda: 80.0)

    with pytest.raises(RuntimeError, match="requires 112.0 GiB"):
        probe_fem_rooms._validate_parallel_resources(
            room_workers=2,
            solver_threads=12,
            minimum_memory_gib_per_worker=56.0,
        )


def test_parallel_resource_gate_accepts_bounded_two_room_launch(monkeypatch):
    monkeypatch.setattr(probe_fem_rooms.os, "cpu_count", lambda: 48)
    monkeypatch.setattr(probe_fem_rooms, "_available_memory_gib", lambda: 149.0)

    probe_fem_rooms._validate_parallel_resources(
        room_workers=2,
        solver_threads=12,
        minimum_memory_gib_per_worker=56.0,
    )
