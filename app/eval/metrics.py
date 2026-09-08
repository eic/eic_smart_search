"""Retrieval metrics + uncertainty, shared by the golden-set runner and the
configuration sweep.

Metric choice follows standard IR practice (TREC-style): a rank-aware graded
measure (nDCG), a coverage measure (recall@k), and a "did the user see it at
all" measure (success@k / MRR). We report every headline number with a
bootstrap 95% confidence interval and compare configurations with a *paired*
bootstrap test, because a 64-question benchmark is small enough that a 3-point
difference is usually noise.

Relevance is binary: a golden case lists the page(s) a correct answer may come
from — they are *alternatives*, not a set that must all be retrieved — so a
citation counts as relevant iff its URL contains any of those needles.

Because the judgments are alternatives, "recall" is undefined here; the
coverage measure is precision@k (how much of the top-k is on-target) and the
ideal ranking used for nDCG is a top-k made entirely of relevant chunks.
"""
from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260812  # fixed → reruns of the same numbers give the same CI


def _relevance(urls: Sequence[str], needles: Sequence[str]) -> list[int]:
    """1 per retrieved position if that URL matches any expected needle."""
    lowered = [n.lower() for n in needles if n]
    return [1 if any(n in (u or "").lower() for n in lowered) else 0 for u in urls]


def rank_metrics(
    urls: Sequence[str], needles: Sequence[str], ks: Iterable[int] = DEFAULT_KS
) -> dict[str, Any]:
    """Per-case retrieval metrics for one ranked citation list.

    Returns {} when the case has no expected URL (abstention / negative cases).
    """
    needles = [n for n in needles if n]
    if not needles:
        return {}
    rel = _relevance(urls, needles)
    rank = next((i + 1 for i, r in enumerate(rel) if r), None)
    out: dict[str, Any] = {
        "hit": rank is not None,
        "rank": rank,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
    }
    for k in ks:
        window = rel[:k]
        out[f"success@{k}"] = 1.0 if any(window) else 0.0
        out[f"precision@{k}"] = round(sum(window) / k, 4)
        out[f"ndcg@{k}"] = round(_ndcg(window, k), 4)
    return out


def _ndcg(rel: Sequence[int], k: int) -> float:
    """Binary-relevance nDCG against an all-relevant ideal top-k.

    Denominators are always k — never the number of results a particular
    configuration happened to return — so a config that returns three sources
    cannot look better than one that returns five good ones simply by having
    fewer slots to fill. The score rewards putting the right source first and
    not padding the list with off-topic chunks.
    """
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg else 0.0


def mean_ci(values: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any] | None:
    """Mean with a bootstrap 95% CI — `None` for an empty sample.

    With ~60 questions the CI is roughly ±0.10 on a rate near 0.5, which is
    exactly the point: it stops us reading a 2-point move as an improvement.
    """
    vals = [float(v) for v in values]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": round(mean, 4), "lo": round(mean, 4), "hi": round(mean, 4), "n": 1}
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(vals)
    means = []
    for _ in range(samples):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n}


def paired_bootstrap_p(
    baseline: Sequence[float], variant: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES
) -> float | None:
    """Two-sided p-value for variant − baseline on *paired* per-question scores.

    Resamples questions (not scores) with replacement, which is the standard
    IR significance test and correctly accounts for the fact that both
    configurations answered the same questions.
    """
    pairs = [(float(b), float(v)) for b, v in zip(baseline, variant)]
    if len(pairs) < 2:
        return None
    observed = sum(v - b for b, v in pairs) / len(pairs)
    if observed == 0:
        return 1.0
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(pairs)
    # Center the differences under H0 (mean difference = 0), then count how
    # often a resample reaches the observed effect by chance.
    diffs = [v - b - observed for b, v in pairs]
    extreme = 0
    for _ in range(samples):
        resampled = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        if abs(resampled) >= abs(observed):
            extreme += 1
    return round((extreme + 1) / (samples + 1), 4)


def stars(p: float | None) -> str:
    """Conventional significance markers for a results table."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""
