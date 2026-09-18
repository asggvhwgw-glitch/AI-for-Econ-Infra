from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_is_project_entrypoint_not_document_dump():
    legacy_root_docs = {
        "ARCHITECTURE.md", "BENCHMARKS.md", "EXTERNAL_VALIDATION.md", "MIGRATION.md",
        "PERFORMANCE_RELEASE.md", "RELEASE_CHECKLIST.md", "RELEASE_MAINTENANCE.json",
        "RELEASE_MANIFEST.md", "TEST_STATUS.md", "UPSTREAM_REFERENCES.md", "VERSIONING.md",
        "bench_hdfe_exact_rank.json",
    }
    assert not any((ROOT / name).exists() for name in legacy_root_docs)
    for name in ("README.md", "CHANGELOG.md", "LICENSE", "NOTICE.md", "pyproject.toml"):
        assert (ROOT / name).is_file()


def test_hdfe_technical_chain_is_canonical():
    tech = ROOT / "docs" / "technical" / "hdfe"
    assert (tech / "README.md").is_file()
    assert (tech / "rank-backends.md").is_file()
    assert (tech / "solver" / "design.md").is_file()
    assert (tech / "solver" / "solver-v0.4.4.md").is_file()
    manuscript = tech / "exact-multiway-dof"
    assert (manuscript / "README.md").is_file()
    assert (manuscript / "exact_multiway_hdfe_dof.tex").stat().st_size > 10_000
    assert (manuscript / "exact_multiway_hdfe_dof.pdf").stat().st_size > 100_000
    numcore = tech / "numerical-residual-core"
    assert (numcore / "README.md").is_file()
    assert (numcore / "exact_multiway_hdfe_residual_core.tex").stat().st_size > 5_000
    assert (numcore / "exact_multiway_hdfe_residual_core.pdf").stat().st_size > 50_000
    structural = ROOT / "docs" / "technical" / "structural-design"
    assert (structural / "README.md").is_file()
    assert (structural / "exact_partition_refinement_hdfe_design.tex").stat().st_size > 5_000
    assert (structural / "exact_partition_refinement_hdfe_design.pdf").stat().st_size > 50_000
    assert (ROOT / "docs" / "technical" / "innovation-audit.md").is_file()
    assert (ROOT / "docs" / "technical" / "innovation-registry.json").is_file()
    assert (ROOT / "benchmarks" / "hdfe" / "exact_rank.json").is_file()
    assert (ROOT / "benchmarks" / "hdfe" / "solver_v044_integration.json").is_file()


def test_release_governance_uses_nested_canonical_manifest():
    assert (ROOT / "docs" / "release" / "maintenance.json").is_file()
    assert not (ROOT / "RELEASE_MAINTENANCE.json").exists()


def test_generated_architecture_map_is_in_development_docs():
    amap = ROOT / "docs" / "development" / "architecture-map"
    assert (amap / "architecture.json").is_file()
    assert (amap / "architecture.md").is_file()
    assert (amap / "architecture.html").is_file()
    assert (ROOT / "skills" / "econhdfe" / "references" / "architecture-visualization.md").is_file()
    assert (ROOT / "skills" / "econhdfe" / "scripts" / "architecture_map.py").is_file()
