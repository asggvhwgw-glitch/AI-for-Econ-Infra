from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .errors import SpecificationError


_MAX_INTERACTION_ORDER = 8
_MAX_EXPANDED_TERMS = 4096


@dataclass(frozen=True, slots=True)
class _FVAtom:
    kind: str  # "factor" | "continuous"
    variable: str
    base_mode: str = "first"  # first | last | freq | none | exact
    base_value: Any | None = None


@dataclass(frozen=True, slots=True)
class _FVTerm:
    atoms: tuple[_FVAtom, ...]


@dataclass(frozen=True, slots=True)
class FactorVariableExpression:
    """Parsed Python-safe factor-variable expression.

    Construct with :func:`fv`; users normally do not instantiate this class
    directly.  The expression is compiled to the existing ``Factor`` and
    ``RegressorInteraction`` design terms by :mod:`econhdfe.design`.
    """

    expression: str
    terms: tuple[_FVTerm, ...]


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    value: str
    pos: int


_NUMBER_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _syntax(message: str, expression: str, pos: int) -> SpecificationError:
    pos = max(0, min(int(pos), len(expression)))
    left = max(0, pos - 28)
    right = min(len(expression), pos + 28)
    excerpt = expression[left:right]
    pointer = " " * (pos - left) + "^"
    return SpecificationError(
        message,
        code="design.factorvars.syntax",
        stage="design",
        details={"expression": expression, "position": pos, "excerpt": excerpt, "pointer": pointer},
        suggestion=(
            "Use Python-safe factor syntax such as "
            "fv('i(x)##c(z)') or fv('i(group, base=last)#c(age)')."
        ),
    )


def _tokenize(expression: str) -> tuple[_Token, ...]:
    out: list[_Token] = []
    i = 0
    n = len(expression)
    while i < n:
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue
        if expression.startswith("##", i):
            out.append(_Token("FULL", "##", i)); i += 2; continue
        if ch == "#":
            out.append(_Token("CROSS", "#", i)); i += 1; continue
        if ch == "+":
            out.append(_Token("PLUS", "+", i)); i += 1; continue
        if ch == "(":
            out.append(_Token("LPAREN", ch, i)); i += 1; continue
        if ch == ")":
            out.append(_Token("RPAREN", ch, i)); i += 1; continue
        if ch == ",":
            out.append(_Token("COMMA", ch, i)); i += 1; continue
        if ch == "=":
            out.append(_Token("EQUAL", ch, i)); i += 1; continue
        if ch in {"'", '"'}:
            quote = ch
            start = i
            i += 1
            chars: list[str] = []
            while i < n:
                ch2 = expression[i]
                if ch2 == "\\":
                    i += 1
                    if i >= n:
                        raise _syntax("unterminated escape in quoted column/base value", expression, start)
                    esc = expression[i]
                    chars.append({"n": "\n", "t": "\t", "r": "\r"}.get(esc, esc))
                    i += 1
                    continue
                if ch2 == quote:
                    i += 1
                    break
                chars.append(ch2)
                i += 1
            else:
                raise _syntax("unterminated quoted string", expression, start)
            out.append(_Token("STRING", "".join(chars), start))
            continue
        m = _NUMBER_RE.match(expression, i)
        if m:
            out.append(_Token("NUMBER", m.group(0), i)); i = m.end(); continue
        m = _IDENT_RE.match(expression, i)
        if m:
            out.append(_Token("IDENT", m.group(0), i)); i = m.end(); continue
        if ch == ".":
            raise _syntax(
                "dot-style factor syntax is intentionally unsupported; write i(x) / c(x), not i.x / c.x",
                expression,
                i,
            )
        raise _syntax(f"unexpected character {ch!r}", expression, i)
    out.append(_Token("EOF", "", n))
    return tuple(out)


def _atom_key(atom: _FVAtom):
    return (atom.kind, atom.variable, atom.base_mode, repr(atom.base_value))


def _term_key(term: _FVTerm):
    # Interactions are commutative, but repeated atoms (c(x)#c(x)) matter.
    return tuple(sorted((_atom_key(a) for a in term.atoms), key=repr))


def _union(*groups: tuple[_FVTerm, ...]) -> tuple[_FVTerm, ...]:
    out: list[_FVTerm] = []
    seen = set()
    for group in groups:
        for term in group:
            key = _term_key(term)
            if key not in seen:
                seen.add(key)
                out.append(term)
                if len(out) > _MAX_EXPANDED_TERMS:
                    raise SpecificationError(
                        "factor-variable expansion is too large",
                        code="design.factorvars.expansion_limit",
                        stage="design",
                        details={"max_terms": _MAX_EXPANDED_TERMS},
                        suggestion="Split the expression or replace very high-cardinality/full-factorial terms with absorbed fixed effects.",
                    )
    return tuple(out)


def _cross(left: tuple[_FVTerm, ...], right: tuple[_FVTerm, ...]) -> tuple[_FVTerm, ...]:
    out: list[_FVTerm] = []
    for a in left:
        for b in right:
            atoms = a.atoms + b.atoms
            if len(atoms) > _MAX_INTERACTION_ORDER:
                raise SpecificationError(
                    f"factor-variable interactions support at most {_MAX_INTERACTION_ORDER} components",
                    code="design.factorvars.interaction_order",
                    stage="design",
                    details={"order": len(atoms), "max_order": _MAX_INTERACTION_ORDER},
                )
            out.append(_FVTerm(atoms))
    return _union(tuple(out))


def _full(left: tuple[_FVTerm, ...], right: tuple[_FVTerm, ...]) -> tuple[_FVTerm, ...]:
    return _union(left, right, _cross(left, right))


class _Parser:
    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = _tokenize(expression)
        self.i = 0

    @property
    def tok(self) -> _Token:
        return self.tokens[self.i]

    def take(self, kind: str) -> _Token:
        tok = self.tok
        if tok.kind != kind:
            raise _syntax(f"expected {kind.lower()}, found {tok.value!r}", self.expression, tok.pos)
        self.i += 1
        return tok

    def maybe(self, kind: str) -> bool:
        if self.tok.kind == kind:
            self.i += 1
            return True
        return False

    def parse(self) -> tuple[_FVTerm, ...]:
        if self.tok.kind == "EOF":
            raise _syntax("factor-variable expression is empty", self.expression, 0)
        out = self.parse_sum()
        if self.tok.kind != "EOF":
            raise _syntax(f"unexpected token {self.tok.value!r}", self.expression, self.tok.pos)
        return out

    def parse_sum(self) -> tuple[_FVTerm, ...]:
        out = self.parse_full()
        while self.maybe("PLUS"):
            out = _union(out, self.parse_full())
        return out

    def parse_full(self) -> tuple[_FVTerm, ...]:
        out = self.parse_cross()
        while self.maybe("FULL"):
            out = _full(out, self.parse_cross())
        return out

    def parse_cross(self) -> tuple[_FVTerm, ...]:
        out = self.parse_atom()
        while self.maybe("CROSS"):
            out = _cross(out, self.parse_atom())
        return out

    def parse_atom(self) -> tuple[_FVTerm, ...]:
        if self.maybe("LPAREN"):
            out = self.parse_sum()
            self.take("RPAREN")
            return out
        if self.tok.kind != "IDENT":
            raise _syntax("expected i(...), c(...), or a parenthesized expression", self.expression, self.tok.pos)
        func = self.take("IDENT")
        if func.value not in {"i", "c"}:
            raise _syntax(f"unknown factor-variable function {func.value!r}; expected i(...) or c(...)", self.expression, func.pos)
        self.take("LPAREN")
        columns: list[str] = [self.parse_column()]
        base_mode = "first"
        base_value: Any | None = None
        while self.maybe("COMMA"):
            if (self.tok.kind == "IDENT" and self.tok.value == "base"
                    and self.tokens[self.i + 1].kind == "EQUAL"):
                self.take("IDENT")
                self.take("EQUAL")
                base_mode, base_value = self.parse_base()
                if self.tok.kind == "COMMA":
                    raise _syntax("base=... must be the final argument", self.expression, self.tok.pos)
                break
            columns.append(self.parse_column())
        self.take("RPAREN")
        if func.value == "c":
            if base_mode != "first" or base_value is not None:
                raise _syntax("c(...) does not accept base=...", self.expression, func.pos)
            return tuple(_FVTerm((_FVAtom("continuous", col),)) for col in columns)
        if len(columns) > 1 and (base_mode != "first" or base_value is not None):
            raise _syntax("base=... with i(...) requires exactly one column", self.expression, func.pos)
        return tuple(_FVTerm((_FVAtom("factor", col, base_mode, base_value),)) for col in columns)

    def parse_column(self) -> str:
        tok = self.tok
        if tok.kind not in {"IDENT", "STRING"}:
            raise _syntax("expected a column name", self.expression, tok.pos)
        self.i += 1
        if not tok.value:
            raise _syntax("column name cannot be empty", self.expression, tok.pos)
        return tok.value

    def parse_base(self) -> tuple[str, Any | None]:
        tok = self.tok
        if tok.kind == "NUMBER":
            self.i += 1
            text = tok.value
            value: Any = float(text) if any(ch in text for ch in ".eE") else int(text)
            return "exact", value
        if tok.kind == "STRING":
            self.i += 1
            return "exact", tok.value
        if tok.kind == "IDENT":
            self.i += 1
            if tok.value in {"first", "last", "freq", "none"}:
                return tok.value, None
            # A bare identifier is a convenient exact base for string-valued factors.
            return "exact", tok.value
        raise _syntax("expected base value, first, last, freq, or none", self.expression, tok.pos)


def fv(expression: str) -> FactorVariableExpression:
    """Parse a Python-safe Stata-like factor-variable expression.

    Examples
    --------
    ``fv("i(group)##c(age)")``
        Main effects plus the group-by-age interaction.
    ``fv("i(a)##i(b)##c(z)")``
        Three-way full factorial.
    ``fv("i(a)##(i(b)+c(z))")``
        Parenthesized/distributive expansion; ``+`` joins terms.
    ``fv("i(group, base=last)#c(age)")``
        Interaction with an explicit reference level.

    The DSL intentionally uses ``i(x)``/``c(x)`` rather than Stata's
    ``i.x``/``c.x`` so the expression remains visually distinct from Python
    attribute access. ``#`` means interaction-only and ``##`` means full
    factorial. Up to eight-way interactions are accepted, matching Stata's
    factor-variable interaction limit.
    """
    if not isinstance(expression, str):
        raise TypeError("fv() expression must be a string")
    terms = _Parser(expression).parse()
    # Stable main-effects-first ordering makes full factorials predictable and
    # mirrors the conventional presentation of A##B##C as A+B+C+AB+AC+BC+ABC.
    terms = tuple(sorted(terms, key=lambda term: len(term.atoms)))
    return FactorVariableExpression(expression=expression, terms=terms)
