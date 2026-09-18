# econhdfe Release Checklist

Review state is `docs/release/maintenance.json`; actual execution state is `docs/release/execution.json`. `scripts/version.py gate` enforces review/compatibility only. Formal authorization additionally requires `scripts/release_acceptance.py check --mode release` with detached evidence and final artifact hashes. See [acceptance](acceptance.md) and the [unified TODO](../../TODO.md).

1. Choose the smallest valid tier using `python scripts/version.py next revision|patch|minor|major`, then bump through `python scripts/version.py bump <version>`; never edit only one version field. REVISION is limited to maintenance-scale changes and cannot carry a public-contract diff.
2. Review public frontend/API and automated compatibility diff.
3. Review backend/compute, errors, result/config schemas, cache/state lifecycle, linked modules, and skill references.
4. Review tests, validation claims, representative performance evidence, dependencies, packaging, migration notes, and release artifacts.
5. Record each area with `scripts/release_maintenance.py set ...`; do not mark external work complete when it was not executed.
6. Run `python scripts/version.py gate`.
7. Build through `scripts/build_release.sh`; do not hand-assemble production artifacts.
8. Confirm source/wheel Python identity, standalone-skill identity, public-contract snapshot, clean wheel smoke tests, source-archive test pass, and recursive bundle hashes.
9. Run `python scripts/technical_innovation.py`; confirm every registered technical-innovation manuscript is present in source/release bundle, byte-identical across them, and absent from the wheel.
10. For testing/beta releases, confirm the reference-parity policy and `benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md` are present in both the repository and standalone Skill; real-user-data benchmarks require explicit consent and must not overwrite historical evidence.
11. Confirm `docs/development/ERROR_REPORT_TEMPLATE.md` is present in source/release artifacts, byte-identical to the standalone Skill `references/error-report-template.md`, and remains privacy-minimized/parameter-only: no data reconstruction, exact identifiers/paths, commands/scripts, raw logs/full tracebacks, or attachments. Benchmarking remains a separate explicit-consent workflow.
12. Treat published artifacts as immutable. Any content change after publication requires a new version.

13. Preserve all blocked/failed/not_run execution checks. Re-run code-affected
validation after the runtime/validation fingerprint changes. A passed unit test
using synthetic acceptance records does not count as real CI or installation.
14. Bind the actual repository commit and final wheel/sdist/source/bundle hashes
in a detached manifest; run formal mode only on those immutable artifacts.
Neither candidate construction nor a `reviewed` entry grants publication authority.
