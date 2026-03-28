"""Tests for rl_core.experiments (RunManager, NamespacedLogger) and
updated rl_core.utils.checkpoint (rng_state, capture/restore)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_core.experiments import NamespacedLogger, RunManager
from rl_core.utils.checkpoint import (
    Checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from rl_core.utils.logging import StdoutLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cfg:
    lr: float = 1e-3
    gamma: float = 0.99
    seed: int = 42


class _CapturingLogger:
    """Logger that records all calls for assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, int]] = []
        self.closed = False

    def log(self, metrics: dict, step: int) -> None:
        self.calls.append((dict(metrics), step))

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# NamespacedLogger
# ---------------------------------------------------------------------------


def test_namespaced_logger_routes_algo_keys() -> None:
    inner = _CapturingLogger()
    logger = NamespacedLogger(
        inner,
        algo_keys={"loss", "q_mean"},
        research_keys={"idn_loss"},
    )
    logger.log({"loss": 0.5, "idn_loss": 0.1, "episode": 10}, step=1)
    assert inner.calls[0][0] == {
        "algo/loss": 0.5,
        "research/idn_loss": 0.1,
        "episode": 10,
    }


def test_namespaced_logger_bare_keys_pass_through() -> None:
    inner = _CapturingLogger()
    logger = NamespacedLogger(inner)
    logger.log({"reward": 42.0}, step=5)
    assert inner.calls[0][0] == {"reward": 42.0}


def test_namespaced_logger_rejects_overlapping_keys() -> None:
    with pytest.raises(ValueError, match="both algo_keys and research_keys"):
        NamespacedLogger(
            StdoutLogger(),
            algo_keys={"loss"},
            research_keys={"loss"},
        )


def test_namespaced_logger_close_delegates() -> None:
    inner = _CapturingLogger()
    logger = NamespacedLogger(inner)
    logger.close()
    assert inner.closed


# ---------------------------------------------------------------------------
# RNG capture / restore
# ---------------------------------------------------------------------------


def test_capture_restore_rng_numpy() -> None:
    np.random.seed(0)
    _ = np.random.rand(10)  # advance state
    state = capture_rng_state()
    a = np.random.rand(5)
    restore_rng_state(state)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_capture_restore_rng_torch() -> None:
    torch.manual_seed(0)
    _ = torch.rand(10)
    state = capture_rng_state()
    a = torch.rand(5)
    restore_rng_state(state)
    b = torch.rand(5)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Checkpoint rng_state round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_rng_state_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    rng = capture_rng_state()
    ckpt = Checkpoint(step=10, state_dicts={}, rng_state=rng)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, path)

    loaded = load_checkpoint(path)
    assert loaded.rng_state is not None
    assert loaded.step == 10

    # Restore and check torch RNG reproduces same draws
    torch.manual_seed(99)  # diverge
    _ = torch.rand(3)
    restore_rng_state(loaded.rng_state)
    a = torch.rand(5)
    restore_rng_state(rng)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_checkpoint_without_rng_state_loads_cleanly(tmp_path: Path) -> None:
    ckpt = Checkpoint(step=0, state_dicts={})
    save_checkpoint(ckpt, tmp_path / "ckpt.pt")
    loaded = load_checkpoint(tmp_path / "ckpt.pt")
    assert loaded.rng_state is None


# ---------------------------------------------------------------------------
# RunManager — basic lifecycle
# ---------------------------------------------------------------------------


def test_run_manager_creates_dirs(tmp_path: Path) -> None:
    cfg = _Cfg()
    inner = _CapturingLogger()
    manager = RunManager(cfg, results_dir=tmp_path, logger=inner)
    with manager.run():
        pass
    assert manager.run_dir.exists()
    assert manager.checkpoints_dir.exists()


def test_run_manager_writes_config_json(tmp_path: Path) -> None:
    cfg = _Cfg(lr=2e-4)
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())
    with manager.run():
        pass
    config_path = manager.run_dir / "config.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["lr"] == pytest.approx(2e-4)


def test_run_manager_status_completed(tmp_path: Path) -> None:
    cfg = _Cfg()
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())
    with manager.run():
        pass
    assert manager.is_done()
    status = json.loads((manager.run_dir / "status.json").read_text())
    assert status["status"] == "completed"


def test_run_manager_status_interrupted(tmp_path: Path) -> None:
    cfg = _Cfg()
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())
    with pytest.raises(KeyboardInterrupt), manager.run():
        raise KeyboardInterrupt
    status = json.loads((manager.run_dir / "status.json").read_text())
    assert status["status"] == "interrupted"
    assert not manager.is_done()


def test_run_manager_status_failed(tmp_path: Path) -> None:
    cfg = _Cfg()
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())
    with pytest.raises(RuntimeError), manager.run():
        raise RuntimeError("boom")
    status = json.loads((manager.run_dir / "status.json").read_text())
    assert status["status"] == "failed"


def test_run_manager_closes_logger(tmp_path: Path) -> None:
    inner = _CapturingLogger()
    manager = RunManager(_Cfg(), results_dir=tmp_path, logger=inner)
    with manager.run():
        pass
    assert inner.closed


# ---------------------------------------------------------------------------
# RunManager — idempotency
# ---------------------------------------------------------------------------


def test_run_manager_is_done_blocks_rerun(tmp_path: Path) -> None:
    cfg = _Cfg()
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())
    with manager.run():
        pass
    assert manager.is_done()
    blocked = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())
    with pytest.raises(RuntimeError, match="already completed"), blocked.run():
        pass


def test_run_manager_force_overwrites_completed(tmp_path: Path) -> None:
    cfg = _Cfg()
    with RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger()).run():
        pass
    # Should not raise with force=True
    manager2 = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger(), force=True)
    with manager2.run():
        pass
    assert manager2.is_done()


def test_run_manager_deterministic_run_id(tmp_path: Path) -> None:
    cfg = _Cfg(seed=1)
    id1 = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger()).run_id
    id2 = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger()).run_id
    assert id1 == id2


def test_run_manager_different_configs_different_ids(tmp_path: Path) -> None:
    id1 = RunManager(_Cfg(seed=1), results_dir=tmp_path, logger=_CapturingLogger()).run_id
    id2 = RunManager(_Cfg(seed=2), results_dir=tmp_path, logger=_CapturingLogger()).run_id
    assert id1 != id2


# ---------------------------------------------------------------------------
# RunManager — checkpointing and resume
# ---------------------------------------------------------------------------


def test_run_checkpoint_and_resume(tmp_path: Path) -> None:
    cfg = _Cfg()
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())

    dummy_state = {"net": {"w": torch.tensor([1.0, 2.0])}}
    with manager.run() as run:
        run.checkpoint(step=500, state_dicts=dummy_state)

    # Second manager, same config — resume from checkpoint
    manager2 = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger(), force=True)
    with manager2.run() as run2:
        start_step, ckpt = run2.resume()
        assert start_step == 500
        assert ckpt is not None
        assert torch.allclose(ckpt.state_dicts["net"]["w"], torch.tensor([1.0, 2.0]))


def test_run_resume_returns_zero_when_no_checkpoint(tmp_path: Path) -> None:
    manager = RunManager(_Cfg(), results_dir=tmp_path, logger=_CapturingLogger())
    with manager.run() as run:
        start_step, ckpt = run.resume()
    assert start_step == 0
    assert ckpt is None


def test_run_checkpoint_restores_rng(tmp_path: Path) -> None:
    cfg = _Cfg()
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())

    torch.manual_seed(42)
    _ = torch.rand(10)  # advance RNG
    with manager.run() as run:
        run.checkpoint(step=1, state_dicts={})
        expected = torch.rand(5)  # what we'd draw after the checkpoint

    manager2 = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger(), force=True)
    with manager2.run() as run2:
        run2.resume()  # restores RNG
        actual = torch.rand(5)

    assert torch.allclose(expected, actual)


def test_run_log_delegates_to_logger(tmp_path: Path) -> None:
    inner = _CapturingLogger()
    manager = RunManager(_Cfg(), results_dir=tmp_path, logger=inner)
    with manager.run() as run:
        run.log(step=10, metrics={"loss": 0.3})
    assert len(inner.calls) == 1
    assert inner.calls[0] == ({"loss": 0.3}, 10)


def test_run_latest_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    cfg = _Cfg()
    manager = RunManager(cfg, results_dir=tmp_path, logger=_CapturingLogger())
    with manager.run() as run:
        run.checkpoint(step=100, state_dicts={})
        run.checkpoint(step=200, state_dicts={})
        run.checkpoint(step=150, state_dicts={})  # out-of-order intentional

    # latest_checkpoint should return 200 (lexicographic sort on zero-padded names)
    latest = manager.latest_checkpoint()
    assert latest is not None
    assert "00000200" in latest.name
