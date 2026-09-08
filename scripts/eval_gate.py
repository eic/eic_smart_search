"""CI quality gate: run the retrieval benchmark, fail on regression.

Usage (from CI, cron, or `make eval-ci`):
    python scripts/eval_gate.py --api http://localhost:8000 \
        [--token $ADMIN_TOKEN] [--min-hit 0.95] [--min-mrr 0.85]
        [--min-ndcg 0.55] [--full]

Runs the golden set (retrieval-only by default: deterministic + free) and
exits non-zero when a threshold is breached — wire it as a required check
or a cron alert. `--full` adds generation + judge (costs LLM tokens).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--token", default=os.environ.get("ADMIN_TOKEN", ""))
    parser.add_argument("--min-hit", type=float, default=0.95)
    parser.add_argument("--min-mrr", type=float, default=0.85)
    # nDCG@5 also catches "right page, but buried under noise" regressions
    # that hit_rate and MRR both miss.
    parser.add_argument("--min-ndcg", type=float, default=0.55)
    parser.add_argument("--full", action="store_true", help="generation + judge")
    parser.add_argument("--label", default="ci-gate")
    args = parser.parse_args()

    body = json.dumps(
        {"label": args.label, "generate": args.full, "judge": args.full}
    ).encode()
    request = urllib.request.Request(
        f"{args.api.rstrip('/')}/eval/run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if args.token:
        request.add_header("Authorization", f"Bearer {args.token}")
    with urllib.request.urlopen(request, timeout=1800) as response:
        run = json.loads(response.read())

    metrics = run.get("metrics") or {}
    hit = metrics.get("hit_rate")
    mrr = metrics.get("mrr")
    ndcg = metrics.get("ndcg@5")
    print(f"eval run {run.get('id')}: hit_rate={hit} mrr={mrr} ndcg@5={ndcg} "
          f"p50={metrics.get('retrieval_latency_ms_p50')}ms")
    for tag, bucket in sorted((metrics.get("by_tag") or {}).items()):
        print(f"  {tag:14} cases={bucket['cases']:>3} hit={bucket.get('hit_rate')}")

    failures: list[str] = []
    if hit is None or hit < args.min_hit:
        failures.append(f"hit_rate {hit} < {args.min_hit}")
    if mrr is None or mrr < args.min_mrr:
        failures.append(f"mrr {mrr} < {args.min_mrr}")
    if ndcg is not None and ndcg < args.min_ndcg:
        failures.append(f"ndcg@5 {ndcg} < {args.min_ndcg}")
    if args.full:
        expected = metrics.get("answer_expected_rate")
        if expected is not None and expected < 0.85:
            failures.append(f"answer_expected_rate {expected} < 0.85")
    if failures:
        print("GATE FAILED: " + "; ".join(failures))
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
