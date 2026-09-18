from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ..config import ExecutionConfig


@dataclass(slots=True)
class ProfileRecorder:
    mode: str = "off"
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (perf_counter() - t0)

    def as_dict(self) -> dict[str, float]:
        return dict(self.timings)


@dataclass(slots=True)
class ExecutionContext:
    """Internal lifecycle container for caches, resource policy and profiling."""
    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    cache: dict[str, Any] = field(default_factory=dict)
    profiler: ProfileRecorder = field(init=False)

    def __post_init__(self):
        self.config.validate()
        self.profiler = ProfileRecorder(self.config.profile)

    def get(self, signature: str):
        if self.config.cache == "off":
            return None
        return self.cache.get(signature)

    def put(self, signature: str, value: Any) -> None:
        if self.config.cache != "off":
            self.cache[signature] = value

    def clear(self) -> None:
        self.cache.clear()
