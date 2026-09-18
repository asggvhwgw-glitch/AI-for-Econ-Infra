# econhdfe Versioning Policy

`econhdfe` uses a PEP 440-compatible four-tier release scheme:

`MAJOR.MINOR.PATCH[.REVISION]`

This is intentionally **not strict Semantic Versioning**, because strict SemVer has only three numeric release components. The optional fourth component gives the project a smaller maintenance serial for changes that do not justify consuming the next patch number.

## Release tiers

### MAJOR

`MAJOR` is reserved for stable-line breaking changes after 1.0, or an equivalent project-wide compatibility reset.

### MINOR

`MINOR` marks a major capability or public-contract milestone. While the package remains pre-1.0, this includes a new estimator family, a materially new user workflow, or a substantial API/architecture milestone that deserves its own release theme.

Examples: `0.4.0 -> 0.5.0`.

### PATCH

`PATCH` is for backward-compatible runtime work that is material enough to matter to normal users. Typical PATCH changes include:

- numerical solver or algorithm changes;
- meaningful performance/cache/execution changes;
- user-visible bug fixes on ordinary estimation paths;
- new backward-compatible public parameters or functions;
- changes to result/config schemas, defaults, or stable structured-error contracts;
- a group of related fixes large enough to deserve a normal release note and validation cycle.

Examples: `0.4.4 -> 0.4.5`.

### REVISION

`REVISION` is the optional fourth numeric component and is used only for maintenance-scale changes on an existing patch base.

Typical REVISION changes include:

- documentation, file-layout, packaging-layout, release-tooling, CI, test, or skill-only maintenance;
- typo/metadata/build fixes;
- extremely localized implementation bug fixes that preserve public API/schema/defaults and do not change estimator definitions or broad numerical strategy;
- an edge-case correctness fix whose effect is confined to the bug-triggering case and does not require a new public contract.

Examples:

- `0.4.4 -> 0.4.4.1`
- `0.4.4.1 -> 0.4.4.2`

A REVISION is a real public release, not a pre-release marker. Pre-releases still use PEP 440 suffixes such as `aN`, `bN`, and `rcN`; for example `0.4.4.1rc1`.

REVISION releases are intentionally constrained by the release gate: they **may not contain a public-contract diff**. If a change adds or alters public exports/signatures, result/config/dataclass schemas, or stable structured-error codes/stages, use PATCH or higher even if the code diff is small.

## Decision rule

Use the smallest tier that truthfully represents user-visible impact:

| Change | Minimum tier |
| --- | --- |
| Docs/file organization/release scripts/tests only | REVISION |
| Very small isolated internal bug, no public-contract change | REVISION |
| Normal user-visible runtime bug fix | PATCH |
| Solver/performance/numerical strategy change | PATCH |
| New backward-compatible public parameter/API/schema/error contract | PATCH |
| New estimator family or material workflow/API milestone | MINOR |
| Stable-line breaking compatibility reset | MAJOR |

When uncertain between REVISION and PATCH, use PATCH if the change can alter ordinary empirical results, estimator semantics, default behavior, or the public contract.

## Current 0.4 line

- `0.4.0`: HDFE computation / exact multiway structural-rank milestone.
- `0.4.1`: repeated linear specification execution/caching.
- `0.4.2`: release-lifecycle gate and public session/frontend error-boundary cleanup.
- `0.4.3`: portable Agent Skill package and public-contract compatibility gate.
- `0.4.4`: backward-compatible multiway HDFE solver-opt2 performance update.
- `0.4.4.1`: technical-documentation and release-layout normalization; no runtime/public API change.
- `0.4.5`: PPML execution-policy/separation-path runtime repair.
- `0.4.6`: advanced one-way cluster inference and weighted-bootstrap repair.
- `0.4.6.1`: package-wide technical-innovation audit, formal manuscript completion and registry/release governance; no runtime/public API change.
- `0.4.7`: corrected adjusted-R-squared fit-DoF semantics and added adjusted within R-squared.
- `0.4.8`: secondary-statistic parity audit, PPML deviance-scale correction, standard ancillary result fields, and testing/beta parity/benchmark workflow.
- `0.4.8.1`: structured error/mismatch-report template and Skill/release feedback workflow; no runtime/public API change.
- `0.4.9`: Python-safe factor-variable expression frontend (`fv`) for linear OLS/IV DataFrame specifications.
- `0.4.10`: role-aware `fv()` integration in absorbed HDFE specifications.
- `0.4.10.1`: privacy-minimized parameter-only error-report/Skill maintenance revision.
- `0.4.10.2`: installed privacy-safe support-report CLI/API with byte-identical packaged templates and release verification.
- `0.5.0`: architecture/performance milestone integrating structured heterogeneous-spec execution, the econometric Data Layer, repeated-workflow encoded datasets, opt-in persistent session resume, the unified Execution Planner, and runtime thread calibration.
- `0.6.0`: identified categorical fixed-effect recovery milestone with solver-independent normalization, realized-sample identification diagnostics, and shared OLS/IV/PPML/IV-PPML post-estimation adapters.

The previously prepared `0.4.5` documentation-layout artifact was reclassified as `0.4.4.1` before external publication so that the history follows this policy. Published artifacts are immutable and are never renumbered retroactively.

Release tags use `vMAJOR.MINOR.PATCH` or `vMAJOR.MINOR.PATCH.REVISION`.

## Version commands

Check the active version:

```bash
python scripts/version.py check
```

Ask the version tool for the next canonical number:

```bash
python scripts/version.py next revision
python scripts/version.py next patch
python scripts/version.py next minor
python scripts/version.py next major
```

Then bump through the controlled entry point:

```bash
python scripts/version.py bump <new-version>
```

REVISION bumps must stay on the current `MAJOR.MINOR.PATCH` base and are sequential. For example, `0.4.4 -> 0.4.4.1 -> 0.4.4.2`. Starting a new patch line uses the three-part form, e.g. `0.4.4.2 -> 0.4.5`.

Local-version suffixes such as `+experiment` are never used for public releases.

## Mandatory version-closeout workflow

A version number is not the release gate by itself. Every bump resets `docs/release/maintenance.json` so all release-impact areas must be explicitly reviewed again.

```bash
python scripts/version.py bump <new-version>
python scripts/compatibility.py diff
# PATCH/MINOR releases may explicitly approve intentional public-contract changes:
python scripts/compatibility.py approve <change-id> --reason "migration/compatibility rationale"
python scripts/release_maintenance.py show
python scripts/release_maintenance.py set <area> reviewed "evidence"
# use status=changed when the area required code/docs changes;
# use status=not_applicable only with an explicit reason.
python scripts/version.py gate
bash scripts/build_release.sh
```

For a REVISION release, `scripts/version.py gate` additionally requires the public-contract diff to be empty; approval cannot be used to bypass that restriction.

`version.py check` checks version consistency and manifest shape but permits pending review items during development. `version.py gate` is strict and is invoked by the release builder.

The mandatory impact areas are responsibility-based rather than file-based: public frontend/API, backend/compute, errors, automated public-contract compatibility, skill/agent contract, linked modules, result/reporting schemas, config/defaults, cache/state lifecycle, tests/validation, performance, docs/migration, packaging/dependencies, external validation, and final release artifacts.

## Public-contract baseline discipline

Every released source archive carries `compatibility/baselines/econhdfe-<version>.json`. A later version compares its live public contract against the previous release snapshot; the version bump only resets the approval file and never rewrites the old baseline. Any difference in public exports/signatures, result/config schemas, exported dataclasses, or structured-error codes/stages must have a change-specific approval and rationale before a PATCH/MINOR gate passes. Editing an old baseline to hide a change is not permitted.

The release build freezes the current public-contract snapshot and architecture map after the review/compatibility gate but BEFORE producing either sdist or wheel. This prevents divergent source artifacts. A snapshot is a contract baseline, not evidence that clean installation or remote CI has passed.

## Execution acceptance (0.6.3+)

A version bump also resets `docs/release/execution.json`; prior successful runs
are not copied into the new release. Candidate acceptance permits explicit
blocked/not_run/failed states but validates successful records and their hashes.
Formal authorization additionally requires all fixed required checks to pass,
a recorded commit and hashed final artifacts in a detached record. See
[acceptance.md](acceptance.md). A meaningful runtime performance change uses PATCH,
so the localized PERF-01 integration advances 0.6.2 to 0.6.3.
