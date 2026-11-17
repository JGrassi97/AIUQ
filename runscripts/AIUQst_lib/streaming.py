"""
Bounded-memory collection of model outputs.

Inference scripts used to keep every output field of every step in memory until the end of
the run, so memory grew with run length x number of output variables and long forecasts were
OOM-killed. ``StepBuffer`` writes each step to per-variable files on disk (numpy memmaps) as it
is produced; post-processing then reads one variable at a time, so peak memory depends on the
largest single variable rather than on the whole output.
"""

# Built-in/Generics
import os
import re
import shutil

# Third party
import numpy as np

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


class StepBuffer:
    """Per-variable on-disk buffers with one slot per output step."""

    def __init__(self, directory: str, capacity: int, flush_every: int = 8):
        if capacity < 1:
            raise ValueError("StepBuffer needs a capacity of at least one step")
        self.directory = directory
        self.capacity = int(capacity)
        # Written pages sit in the page cache, which a cgroup-v2 job is charged for. Flushing
        # regularly keeps them clean, so the kernel can drop them instantly under the job's
        # memory limit instead of stalling on write-back.
        self.flush_every = max(1, int(flush_every))
        self.count = 0
        self._maps: dict[str, np.memmap] = {}
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.directory, _SAFE_NAME.sub("_", name) + ".npy")

    def append(self, arrays: dict) -> None:
        """Store one step: ``{name: array}``, all arrays without the step dimension."""
        if self.count >= self.capacity:
            raise IndexError(f"StepBuffer is full ({self.capacity} steps)")
        arrays = {name: np.asarray(value) for name, value in arrays.items()}
        # Validate the whole step before writing anything.
        if self.count:
            new = set(arrays) - set(self._maps)
            if new:
                raise KeyError(f"Variables first appeared at step {self.count}: {sorted(new)}")
            missing = set(self._maps) - set(arrays)
            if missing:
                raise KeyError(f"Variables missing at step {self.count}: {sorted(missing)}")
            for name, value in arrays.items():
                expected = self._maps[name].shape[1:]
                if value.shape != expected:
                    raise ValueError(f"Shape of '{name}' changed: {value.shape} vs {expected}")
        for name, value in arrays.items():
            if name not in self._maps:
                self._maps[name] = np.lib.format.open_memmap(
                    self._path(name), mode="w+", dtype=value.dtype, shape=(self.capacity,) + value.shape)
            self._maps[name][self.count] = value
        self.count += 1
        if self.count % self.flush_every == 0:
            for mapped in self._maps.values():
                mapped.flush()

    @property
    def names(self) -> list[str]:
        return list(self._maps)

    def read(self, name: str) -> np.ndarray:
        """All stored steps of one variable, memory-mapped read-only (shape: steps x ...)."""
        if name not in self._maps:
            raise KeyError(name)
        self._maps[name].flush()
        return np.load(self._path(name), mmap_mode="r")[: self.count]

    def close(self) -> None:
        """Release the memmaps and delete the files."""
        for mapped in self._maps.values():
            del mapped
        self._maps.clear()
        shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
