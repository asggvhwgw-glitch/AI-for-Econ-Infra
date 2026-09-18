from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..errors import InputError


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    code: str
    message: str
    variable: str | None = None
    role: str | None = None
    severity: str = "error"
    suggestion: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(x.severity == "error" for x in self.issues)

    @property
    def errors(self) -> tuple[PreflightIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "error")

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(x for x in self.issues if x.severity != "error")

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        first = self.errors[0]
        raise InputError(
            first.message, code=first.code, stage="frontend",
            details={"variable": first.variable, "role": first.role, **(first.details or {})},
            suggestion=first.suggestion,
        )
