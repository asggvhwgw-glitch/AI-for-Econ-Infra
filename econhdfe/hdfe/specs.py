from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Interaction:
    """Categorical interaction used as one absorbed fixed effect.

    ``interaction("city", "year")`` corresponds to Stata
    ``i.city#i.year`` inside ``absorb()``.  Components are factorized jointly;
    no string-concatenated interaction column needs to be materialized.
    """
    parts: tuple[Any, ...]
    name: str | None = None

    def __init__(self, *parts: Any, name: str | None = None):
        if len(parts) < 2:
            raise ValueError("an interaction requires at least two categorical components")
        object.__setattr__(self, "parts", tuple(parts))
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class FixedEffect:
    """Specification for an absorbed fixed effect.

    ``group`` may be a categorical column/array or an :class:`Interaction`.
    ``slopes`` are continuous variables whose coefficients vary by ``group``.
    """
    group: Any
    slopes: tuple[Any, ...] = ()
    intercept: bool = True
    name: str | None = None

    def __init__(self, group: Any, slopes=(), intercept: bool = True, name: str | None = None):
        object.__setattr__(self, "group", group)
        if slopes is None:
            slopes = ()
        elif isinstance(slopes, (str, bytes)):
            slopes = (slopes,)
        else:
            slopes = tuple(slopes)
        object.__setattr__(self, "slopes", slopes)
        object.__setattr__(self, "intercept", bool(intercept))
        object.__setattr__(self, "name", name)


def fe(group: Any, *slopes: Any, intercept: bool = True, name: str | None = None) -> FixedEffect:
    return FixedEffect(group=group, slopes=slopes, intercept=intercept, name=name)


def interaction(*parts: Any, name: str | None = None) -> Interaction:
    """Compact categorical-interaction helper for absorbed FEs."""
    return Interaction(*parts, name=name)
