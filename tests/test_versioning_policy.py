from __future__ import annotations

import sys
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import version as versioning


def test_public_version_accepts_patch_and_revision_forms():
    assert versioning._validate_public("0.4.4").release == (0, 4, 4)
    assert versioning._validate_public("0.4.4.1").release == (0, 4, 4, 1)
    assert versioning._validate_public("0.4.4.1rc1").release == (0, 4, 4, 1)
    assert versioning.version_tier("0.4.4") == "patch-or-higher"
    assert versioning.version_tier("0.4.4.1") == "revision"


@pytest.mark.parametrize("bad", ["0.4", "0.4.4.0", "0.4.4.1.1", "0.4.4+local"])
def test_public_version_rejects_noncanonical_forms(bad):
    with pytest.raises(SystemExit):
        versioning._validate_public(bad)


def test_next_version_respects_four_tier_policy():
    assert versioning.next_version("revision", "0.4.4") == "0.4.4.1"
    assert versioning.next_version("revision", "0.4.4.1") == "0.4.4.2"
    assert versioning.next_version("patch", "0.4.4.7") == "0.4.5"
    assert versioning.next_version("minor", "0.4.4.7") == "0.5.0"
    assert versioning.next_version("major", "0.4.4.7") == "1.0.0"


def test_revision_transition_stays_on_base_and_is_sequential():
    versioning._validate_transition(Version("0.4.4"), Version("0.4.4.1"))
    versioning._validate_transition(Version("0.4.4.1"), Version("0.4.4.2"))
    versioning._validate_transition(Version("0.4.4.2"), Version("0.4.5"))
    with pytest.raises(SystemExit, match="sequential"):
        versioning._validate_transition(Version("0.4.4"), Version("0.4.4.2"))
    with pytest.raises(SystemExit, match="current MAJOR.MINOR.PATCH base"):
        versioning._validate_transition(Version("0.4.4"), Version("0.4.5.1"))


def test_revision_gate_rejects_public_contract_changes(monkeypatch):
    monkeypatch.setattr(versioning, "check", lambda **kwargs: "0.4.4.1")
    monkeypatch.setattr(versioning, "check_manifest", lambda **kwargs: None)
    monkeypatch.setattr(
        versioning,
        "check_compatibility",
        lambda **kwargs: [{"id": "x", "path": "exports", "kind": "changed"}],
    )
    monkeypatch.setattr(versioning, "check_skill", lambda: None)
    with pytest.raises(SystemExit, match="REVISION releases may not change the public contract"):
        versioning.gate()
