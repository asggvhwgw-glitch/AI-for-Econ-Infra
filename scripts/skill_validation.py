"""Validate the canonical econhdfe Agent Skill package."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "econhdfe"


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.DOTALL)
    if not m:
        raise ValueError("SKILL.md must start with YAML frontmatter")
    fields: dict[str, str] = {}
    for raw in m.group(1).splitlines():
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError(f"invalid frontmatter line: {raw!r}")
        key, value = raw.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError(f"invalid frontmatter line: {raw!r}")
        fields[key] = value.strip('"\'')
    return fields, m.group(2)


def validate_skill(skill_root: Path = SKILL_ROOT) -> list[str]:
    problems: list[str] = []
    skill_md = skill_root / "SKILL.md"
    if not skill_md.exists():
        return [f"missing canonical skill: {skill_md}"]
    text = skill_md.read_text()
    try:
        fields, body = _frontmatter(text)
    except ValueError as exc:
        return [str(exc)]

    if set(fields) != {"name", "description"}:
        problems.append("SKILL.md frontmatter must contain only name and description")
    name = fields.get("name", "")
    if name != "econhdfe":
        problems.append("skill name must be econhdfe")
    if not re.fullmatch(r"[a-z0-9-]{1,64}", name):
        problems.append("skill name violates Agent Skills naming constraints")
    desc = fields.get("description", "")
    if not desc or len(desc) > 1024:
        problems.append("skill description must be non-empty and <=1024 characters")
    if len(text.splitlines()) > 500:
        problems.append("SKILL.md must remain <=500 lines; move conditional detail into references")

    references = sorted((skill_root / "references").glob("*.md"))
    if not references:
        problems.append("skill references directory is empty")
    for ref in references:
        rel = ref.relative_to(skill_root).as_posix()
        if f"({rel})" not in body:
            problems.append(f"reference is not directly linked from SKILL.md: {rel}")

    # Validate all local markdown links in the router.
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        if "://" in target or target.startswith("#"):
            continue
        path = (skill_root / target).resolve()
        try:
            path.relative_to(skill_root.resolve())
        except ValueError:
            problems.append(f"skill link escapes skill root: {target}")
            continue
        if not path.exists():
            problems.append(f"broken skill link: {target}")

    agents_yaml = skill_root / "agents" / "openai.yaml"
    if not agents_yaml.exists():
        problems.append("missing agents/openai.yaml")
    else:
        y = agents_yaml.read_text()
        m_short = re.search(r'^\s*short_description:\s*"([^"]*)"\s*$', y, re.MULTILINE)
        m_prompt = re.search(r'^\s*default_prompt:\s*"([^"]*)"\s*$', y, re.MULTILINE)
        if not re.search(r'^\s*display_name:\s*"[^"]+"\s*$', y, re.MULTILINE):
            problems.append("agents/openai.yaml missing display_name")
        if not m_short or not (25 <= len(m_short.group(1)) <= 64):
            problems.append("agents/openai.yaml short_description must be 25-64 characters")
        if not m_prompt or "$econhdfe" not in m_prompt.group(1):
            problems.append("agents/openai.yaml default_prompt must explicitly mention $econhdfe")

    for script in sorted((skill_root / "scripts").glob("*.py")):
        try:
            compile(script.read_text(), str(script), "exec")
        except SyntaxError as exc:
            problems.append(f"skill script syntax error in {script.name}: {exc}")

    # Legacy duplicate entrypoints caused version/content drift in <=0.4.2.
    if (ROOT / "SKILL.md").exists():
        problems.append("legacy root SKILL.md must not be restored; skills/econhdfe is canonical")
    if (ROOT / "skill" / "SKILL.md").exists():
        problems.append("legacy skill/SKILL.md must not be restored; skills/econhdfe is canonical")
    return problems


def check_skill(skill_root: Path = SKILL_ROOT) -> None:
    problems = validate_skill(skill_root)
    if problems:
        raise SystemExit("skill validation failed:\n- " + "\n- ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-root", type=Path, default=SKILL_ROOT)
    args = ap.parse_args()
    check_skill(args.skill_root.resolve())
    print(f"skill validation PASS: {args.skill_root}")


if __name__ == "__main__":
    main()
