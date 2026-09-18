"""Machine-readable release closeout checks for econhdfe.

The manifest is deliberately simple JSON so the release gate works on every
supported Python version without adding a build dependency.  A version bump
resets every required review area to ``pending``; the release builder refuses
to publish until each area is explicitly reviewed.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "release" / "maintenance.json"
SCHEMA_VERSION = 1

# Keep this list compact and responsibility-oriented.  New checks should be
# added only when they represent a distinct release contract, not a one-off bug.
REQUIRED_CHECKS = OrderedDict([
    ("frontend_api", "Public call signatures, exports, aliases, CLI/front-end validation and compatibility."),
    ("backend_compute", "Solver/compute/cache/backend contracts, numerical defaults and resource controls."),
    ("errors", "Structured exception classes, codes, stages, suggestions and public error boundaries."),
    ("public_contract", "Public exports/signatures, result/config schemas and stable structured-error compatibility."),
    ("skill", "Agent skill commands, examples, decision rules and newly exposed controls."),
    ("linked_modules", "Cross-module imports, shared schemas, adapters, compatibility shims and downstream consumers."),
    ("results_reporting", "Result fields, publication output, diagnostics metadata and serialization-facing schemas."),
    ("config_defaults", "Config objects, defaults, deprecations, aliases and backwards-compatible behavior."),
    ("cache_state", "Reusable state/cache invalidation, fingerprints, lifecycle semantics and stale-state hazards."),
    ("tests_validation", "Regression tests, install smoke tests, golden/parity coverage and failure-path tests."),
    ("performance", "Representative benchmark/resource regressions for code paths materially changed by the release."),
    ("docs_migration", "README/architecture/changelog/migration/versioning documentation and user-visible examples."),
    ("packaging_dependencies", "Python/dependency matrix, optional extras, wheel/source contents and build metadata."),
    ("external_validation", "Licensed/reference-environment parity gates and explicitly deferred external certification."),
    ("release_artifacts", "Version consistency, source/wheel parity, hashes, clean install and reproducible bundle contents."),
])

COMPLETE_STATUSES = {"reviewed", "changed", "not_applicable"}
ALL_STATUSES = COMPLETE_STATUSES | {"pending"}


def new_manifest(version: str, previous_version: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "release_version": str(version),
        "previous_version": str(previous_version),
        "checks": {
            name: {"status": "pending", "evidence": "", "description": description}
            for name, description in REQUIRED_CHECKS.items()
        },
    }


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"release maintenance manifest missing: {path.name}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid release maintenance manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("release maintenance manifest must contain a JSON object")
    return data


def write_manifest(data: dict[str, Any], path: Path = MANIFEST) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def reset_manifest(version: str, previous_version: str, path: Path = MANIFEST) -> dict[str, Any]:
    data = new_manifest(version, previous_version)
    write_manifest(data, path)
    return data


def _evidence_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(isinstance(x, str) and x.strip() for x in value)
    return False


def validate_manifest(
    data: dict[str, Any], *, expected_version: str | None = None, require_complete: bool = True
) -> list[str]:
    problems: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if expected_version is not None and data.get("release_version") != expected_version:
        problems.append(
            f"release_version mismatch: manifest={data.get('release_version')!r}, expected={expected_version!r}"
        )
    checks = data.get("checks")
    if not isinstance(checks, dict):
        return problems + ["checks must be an object"]
    for name in REQUIRED_CHECKS:
        item = checks.get(name)
        if not isinstance(item, dict):
            problems.append(f"missing check: {name}")
            continue
        status = item.get("status")
        if status not in ALL_STATUSES:
            problems.append(f"{name}: invalid status {status!r}")
            continue
        if require_complete and status == "pending":
            problems.append(f"{name}: pending")
        if status in COMPLETE_STATUSES and not _evidence_present(item.get("evidence")):
            problems.append(f"{name}: completed checks require non-empty evidence")
    return problems


def check_manifest(*, expected_version: str | None = None, require_complete: bool = True) -> dict[str, Any]:
    data = load_manifest()
    problems = validate_manifest(data, expected_version=expected_version, require_complete=require_complete)
    if problems:
        raise SystemExit("release maintenance gate failed:\n- " + "\n- ".join(problems))
    return data


def set_check(name: str, status: str, evidence: str) -> None:
    if name not in REQUIRED_CHECKS:
        raise SystemExit(f"unknown release maintenance check: {name}")
    if status not in COMPLETE_STATUSES:
        raise SystemExit("status must be reviewed, changed, or not_applicable")
    if not evidence.strip():
        raise SystemExit("evidence must be non-empty")
    data = load_manifest()
    data["checks"][name] = {
        "status": status,
        "evidence": evidence.strip(),
        "description": REQUIRED_CHECKS[name],
    }
    write_manifest(data)


def _summary(data: dict[str, Any]) -> str:
    checks = data.get("checks", {})
    done = sum(isinstance(v, dict) and v.get("status") in COMPLETE_STATUSES for v in checks.values())
    return f"release maintenance: {done}/{len(REQUIRED_CHECKS)} complete for {data.get('release_version')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("reset")
    rp.add_argument("version")
    rp.add_argument("--previous", required=True)
    cp = sub.add_parser("check")
    cp.add_argument("--version")
    cp.add_argument("--allow-pending", action="store_true")
    sp = sub.add_parser("set")
    sp.add_argument("name", choices=tuple(REQUIRED_CHECKS))
    sp.add_argument("status", choices=tuple(sorted(COMPLETE_STATUSES)))
    sp.add_argument("evidence")
    sub.add_parser("show")
    args = ap.parse_args()

    if args.command == "reset":
        data = reset_manifest(args.version, args.previous)
        print(_summary(data))
    elif args.command == "check":
        data = check_manifest(expected_version=args.version, require_complete=not args.allow_pending)
        print(_summary(data))
    elif args.command == "set":
        set_check(args.name, args.status, args.evidence)
        print(_summary(load_manifest()))
    else:
        data = load_manifest()
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
