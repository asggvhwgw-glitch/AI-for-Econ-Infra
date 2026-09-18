"""Validate the package-wide technical-innovation registry and manuscript chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "technical" / "innovation-registry.json"
AUDIT = ROOT / "docs" / "technical" / "innovation-audit.md"


def load_registry(root: Path = ROOT) -> dict:
    path = root / REGISTRY.relative_to(ROOT)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing innovation registry: {path}") from exc
    if data.get("schema_version") != 1:
        raise ValueError("unsupported innovation-registry schema_version")
    return data


def validate_registry(root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    registry = root / "docs" / "technical" / "innovation-registry.json"
    audit = root / "docs" / "technical" / "innovation-audit.md"
    if not registry.is_file():
        return [f"missing innovation registry: {registry}"]
    if not audit.is_file():
        problems.append(f"missing innovation audit: {audit}")
    try:
        data = json.loads(registry.read_text())
    except Exception as exc:  # release validator: report malformed JSON cleanly
        return [f"invalid innovation registry JSON: {exc}"]
    if data.get("schema_version") != 1:
        problems.append("innovation registry schema_version must be 1")
    items = data.get("innovations")
    if not isinstance(items, list) or not items:
        problems.append("innovation registry must contain at least one innovation")
        return problems
    ids: set[str] = set()
    manuscripts: set[str] = set()
    for idx, item in enumerate(items):
        prefix = f"innovation[{idx}]"
        iid = str(item.get("id", "")).strip()
        if not iid:
            problems.append(f"{prefix}: missing id")
        elif iid in ids:
            problems.append(f"{prefix}: duplicate id {iid}")
        ids.add(iid)
        if item.get("classification") not in {"theorem_backed_framework", "theorem_backed_application", "technical_innovation"}:
            problems.append(f"{iid or prefix}: invalid mathematical-contribution classification")
        review = item.get("mathematical_review")
        if not isinstance(review, str) or not (root / review).is_file():
            problems.append(f"{iid or prefix}: missing mathematical review")
        if not item.get("originality_status"):
            problems.append(f"{iid or prefix}: missing explicit originality status")
        for field in ("title", "novelty_scope", "prior_art_boundary"):
            if not str(item.get(field, "")).strip():
                problems.append(f"{iid or prefix}: missing {field}")
        manuscript = item.get("manuscript") or {}
        tex_rel = manuscript.get("tex")
        pdf_rel = manuscript.get("pdf")
        for kind, rel, suffix, minimum in (
            ("tex", tex_rel, ".tex", 5_000),
            ("pdf", pdf_rel, ".pdf", 50_000),
        ):
            if not isinstance(rel, str) or not rel.endswith(suffix):
                problems.append(f"{iid or prefix}: invalid manuscript {kind} path")
                continue
            if rel in manuscripts:
                problems.append(f"{iid or prefix}: duplicate manuscript path {rel}")
            manuscripts.add(rel)
            path = root / rel
            if not path.is_file():
                problems.append(f"{iid or prefix}: missing manuscript {kind}: {rel}")
            elif path.stat().st_size <= minimum:
                problems.append(f"{iid or prefix}: manuscript {kind} unexpectedly small: {rel}")
        for field in ("implementation", "tests", "evidence"):
            paths = item.get(field)
            if not isinstance(paths, list) or not paths:
                problems.append(f"{iid or prefix}: {field} must be a non-empty path list")
                continue
            for rel in paths:
                if not isinstance(rel, str) or not rel.strip():
                    problems.append(f"{iid or prefix}: invalid {field} path")
                elif not (root / rel).is_file():
                    problems.append(f"{iid or prefix}: missing {field} path: {rel}")
    return problems


def manuscript_paths(root: Path = ROOT) -> tuple[Path, ...]:
    data = load_registry(root)
    out: list[Path] = []
    for item in data["innovations"]:
        manuscript = item["manuscript"]
        out.extend((Path(manuscript["tex"]), Path(manuscript["pdf"])))
    return tuple(out)


def check_registry(root: Path = ROOT) -> None:
    problems = validate_registry(root)
    if problems:
        raise SystemExit("technical innovation validation failed:\n- " + "\n- ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.root.resolve()
    problems = validate_registry(root)
    if problems:
        raise SystemExit("technical innovation validation failed:\n- " + "\n- ".join(problems))
    data = load_registry(root)
    print(f"technical innovation validation PASS: {len(data['innovations'])} registered mathematical contribution(s)")


if __name__ == "__main__":
    main()
