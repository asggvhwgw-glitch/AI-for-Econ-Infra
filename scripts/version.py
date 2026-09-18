"""Check or bump the econhdfe package version from one controlled entry point."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from packaging.version import Version, InvalidVersion

from release_maintenance import check_manifest, reset_manifest
from release_acceptance import new_manifest, RELATIVE_MANIFEST
import json
from compatibility import check_compatibility, reset_approvals, require_baseline
from skill_validation import check_skill
from technical_innovation import check_registry

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "econhdfe" / "__init__.py"


def _project_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.MULTILINE)
    if not m:
        raise SystemExit("project version not found in pyproject.toml")
    return m.group(1)


def _runtime_version() -> str:
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', INIT.read_text(), re.MULTILINE)
    if not m:
        raise SystemExit("runtime __version__ not found")
    return m.group(1)


def _validate_public(v: str) -> Version:
    try:
        parsed = Version(v)
    except InvalidVersion as exc:
        raise SystemExit(f"invalid PEP 440 version: {v}") from exc
    if parsed.local is not None:
        raise SystemExit("public econhdfe versions must not contain a local '+...' suffix")
    if len(parsed.release) not in {3, 4}:
        raise SystemExit(
            "econhdfe public versions must use MAJOR.MINOR.PATCH or "
            "MAJOR.MINOR.PATCH.REVISION"
        )
    if len(parsed.release) == 4 and parsed.release[3] <= 0:
        raise SystemExit("REVISION must be a positive integer; omit it instead of using .0")
    return parsed


def version_tier(v: str | Version) -> str:
    """Return the public release tier encoded by *v*."""
    parsed = v if isinstance(v, Version) else _validate_public(v)
    return "revision" if len(parsed.release) == 4 else "patch-or-higher"


def next_version(kind: str, current: str | Version) -> str:
    """Compute the next canonical version for a requested release tier."""
    parsed = current if isinstance(current, Version) else _validate_public(current)
    major, minor, patch = parsed.release[:3]
    revision = parsed.release[3] if len(parsed.release) == 4 else 0
    if kind == "revision":
        return f"{major}.{minor}.{patch}.{revision + 1}"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "major":
        return f"{major + 1}.0.0"
    raise SystemExit(f"unknown version tier: {kind}")


def _validate_transition(current: Version, target: Version) -> None:
    if target <= current:
        raise SystemExit(f"new version must be greater than current version {current}")
    if len(target.release) != 4:
        return
    # A maintenance revision stays on one MAJOR.MINOR.PATCH base and increments
    # monotonically. Starting a new patch/minor line must first use a 3-part version.
    if target.release[:3] != current.release[:3]:
        raise SystemExit(
            "a REVISION bump must stay on the current MAJOR.MINOR.PATCH base; "
            "start a new patch/minor line with a 3-part version first"
        )
    current_revision = current.release[3] if len(current.release) == 4 else 0
    target_revision = target.release[3]
    if current.pre is None and target.pre is None and target_revision != current_revision + 1:
        raise SystemExit(
            f"REVISION bumps must be sequential: expected .{current_revision + 1}, "
            f"got .{target_revision}"
        )


def check(*, maintenance: bool = True) -> str:
    a, b = _project_version(), _runtime_version()
    _validate_public(a)
    if a != b:
        raise SystemExit(f"version mismatch: pyproject={a}, runtime={b}")
    if maintenance:
        check_manifest(expected_version=a, require_complete=False)
    return a


def gate() -> str:
    version = check(maintenance=False)
    check_manifest(expected_version=version, require_complete=True)
    changes = check_compatibility(expected_version=version)
    if version_tier(version) == "revision" and changes:
        raise SystemExit(
            "REVISION releases may not change the public contract; "
            "use a PATCH or higher version for public API/schema/error-contract changes"
        )
    check_skill()
    check_registry()
    return version


def bump(v: str) -> None:
    target = _validate_public(v)
    old = check(maintenance=False)
    current = _validate_public(old)
    require_baseline(old)
    _validate_transition(current, target)
    p = PYPROJECT.read_text()
    p = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{v}"', p, count=1, flags=re.MULTILINE)
    PYPROJECT.write_text(p)
    i = INIT.read_text()
    i = re.sub(r'^__version__\s*=\s*["\'][^"\']+["\']', f'__version__ = "{v}"', i, count=1, flags=re.MULTILINE)
    INIT.write_text(i)
    reset_manifest(v, old)
    execution_path = ROOT / RELATIVE_MANIFEST
    execution_path.parent.mkdir(parents=True, exist_ok=True)
    execution_path.write_text(json.dumps(new_manifest(ROOT, v), indent=2) + "\n", encoding="utf-8")
    reset_approvals(baseline_version=old, release_version=v)
    print(f"version: {old} -> {v}")
    print("release maintenance reset: all required review areas are pending")
    print(f"compatibility baseline: {old}; approvals reset for {v}")
    print("complete them with scripts/release_maintenance.py set ... before scripts/version.py gate")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("gate")
    np = sub.add_parser("next")
    np.add_argument("kind", choices=("revision", "patch", "minor", "major"))
    bp = sub.add_parser("bump")
    bp.add_argument("version")
    args = ap.parse_args()
    if args.command == "check":
        print(check())
    elif args.command == "gate":
        print(gate())
    elif args.command == "next":
        print(next_version(args.kind, check(maintenance=False)))
    else:
        bump(args.version)


if __name__ == "__main__":
    main()
