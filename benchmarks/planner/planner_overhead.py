from __future__ import annotations
import json, time
from pathlib import Path
from econhdfe.planner import RuntimeResources, PlanCertificate, RepresentationCandidate, plan_execution

resources = RuntimeResources(8, 32 * 1024**3, 4 * 1024**3)
candidates = (
    RepresentationCandidate('dense', PlanCertificate.exact_yes('dense'), 512 * 1024**2),
    RepresentationCandidate('block_dense', PlanCertificate.exact_yes('block_dense'), 128 * 1024**2, setup_bytes=64 * 1024**2),
)
for _ in range(1000):
    plan_execution(requested_budget_bytes=2*1024**3, requested_threads='auto', resources=resources,
                   representation_candidates=candidates, expected_passes=3)
n = 100_000
t0 = time.perf_counter()
for _ in range(n):
    plan_execution(requested_budget_bytes=2*1024**3, requested_threads='auto', resources=resources,
                   representation_candidates=candidates, expected_passes=3)
elapsed = time.perf_counter() - t0
out = {
    'calls': n,
    'seconds': elapsed,
    'microseconds_per_plan': elapsed / n * 1e6,
    'note': 'Pure planner overhead microbenchmark; excludes specification analysis and estimator work.'
}
Path('benchmarks/planner/planner_overhead.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
