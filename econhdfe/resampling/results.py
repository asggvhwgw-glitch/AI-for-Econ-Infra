from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplicateFailure:
    code: str
    type: str
    message: str


@dataclass(slots=True)
class ReplicateBatch:
    values: list[Any] = field(default_factory=list)
    failures: list[ReplicateFailure] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return len(self.values)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def failure_counts(self) -> dict[str, int]:
        return dict(Counter(x.code for x in self.failures))
