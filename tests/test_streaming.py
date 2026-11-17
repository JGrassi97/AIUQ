"""On-disk per-variable step buffers used by the inference scripts."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
from AIUQst_lib.streaming import StepBuffer  # noqa: E402


def test_steps_round_trip_and_files_are_removed(tmp_path):
    directory = tmp_path / "stream"
    rng = np.random.default_rng(0)
    surface = [rng.random((3, 4), dtype=np.float32) for _ in range(5)]
    levels = [rng.random((2, 3, 4)) for _ in range(5)]
    with StepBuffer(str(directory), capacity=7) as buffer:
        for sfc, pl in zip(surface, levels):
            buffer.append({"t2m": sfc, "t/850": pl})
        assert buffer.count == 5 and sorted(buffer.names) == ["t/850", "t2m"]
        t2m = buffer.read("t2m")
        assert t2m.shape == (5, 3, 4) and t2m.dtype == np.float32
        np.testing.assert_array_equal(t2m, np.stack(surface))
        np.testing.assert_array_equal(buffer.read("t/850"), np.stack(levels))
        assert all("/" not in p.name for p in directory.iterdir())
    assert not directory.exists()


def test_buffer_rejects_inconsistent_steps(tmp_path):
    buffer = StepBuffer(str(tmp_path / "s"), capacity=2)
    buffer.append({"a": np.zeros(3), "b": np.zeros(2)})
    with pytest.raises(ValueError, match="Shape"):
        buffer.append({"a": np.full(3, 9.), "b": np.zeros(4)})
    with pytest.raises(KeyError, match="missing"):
        buffer.append({"a": np.full(3, 9.)})
    with pytest.raises(KeyError, match="first appeared"):
        buffer.append({"a": np.ones(3), "b": np.ones(2), "c": np.ones(1)})
    assert buffer.count == 1                       # rejected steps leave nothing behind
    buffer.append({"a": np.ones(3), "b": np.ones(2)})
    np.testing.assert_array_equal(buffer.read("a"), [np.zeros(3), np.ones(3)])
    with pytest.raises(IndexError, match="full"):
        buffer.append({"a": np.ones(3), "b": np.ones(2)})
    buffer.close()


def test_existing_directory_is_reset(tmp_path):
    directory = tmp_path / "s"
    directory.mkdir()
    (directory / "stale.npy").write_bytes(b"old")
    with StepBuffer(str(directory), capacity=1) as buffer:
        assert list(directory.iterdir()) == []
        buffer.append({"x": np.arange(2)})
        np.testing.assert_array_equal(buffer.read("x"), [[0, 1]])
