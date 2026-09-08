# How good is EIC Smart Search? — Evaluation methodology and results

*A plain-language description of how this product is measured, written for
readers outside the project. Live numbers: [/admin/eval](https://api.aprozo.space/admin/eval).*

Methodology influences: execution-style deterministic pass/fail and
per-category reporting as used in ROOTER (CodeGraph RAG for ROOT,
IHEP — [talk](https://indico.cern.ch/event/1471803/contributions/6968237/)),
plus LLM-as-a-judge grading as deployed by CMS's Archi
([arXiv:2606.04755](https://arxiv.org/abs/2606.04755)).

## What is being measured

The assistant answers questions about the EIC/ePIC ecosystem by searching
~20,000 indexed documents (websites, wikis, GitHub, papers, meeting slide
decks) and generating a cited answer. Two things can go wrong, so two
things are measured **separately**:

1. **Did search find the right source?** — deterministic, no AI judging.
2. **Is the generated answer correct and grounded?** — substring checks
   plus an independent LLM judge.

## The benchmark

A **golden set** of real questions (61+ cases, growing), each with
machine-checkable expectations:

```yaml
- id: lfhcal-layers
  query: "how many layers in lfhcal?"          # exact real-user phrasing
  expect_url_contains: ["Forward_Hcal"]        # the page a correct answer needs
  expect_answer_contains: ["60"]               # the fact the answer must state
```

Three properties make it honest:

- **Real questions.** Cases are mined from the live query log — including
  typos and vague phrasing ("give me TDR", "hot to install eic-shell") —
  and from agents that read the indexed corpus and wrote questions a
  physicist would ask. Expectations were verified against the actual
  documents before inclusion.
- **Failure-driven growth.** Whenever a real query produces a bad answer,
  it becomes a benchmark case *first*, then gets fixed. The benchmark
  ratchets harder over time; today's 100% is against every failure ever
  observed.
- **No contamination.** Evaluation traffic bypasses the response cache
  and analytics (`eval_mode`), so the system cannot memorize its way to a
  score, and users never see eval noise.

## Pass criteria (deterministic first)

| Check | Definition | Style |
|---|---|---|
| Right source found | Any top-8 citation URL contains an expected fragment | deterministic |
| nDCG@5 | Rank-aware quality of the top 5, against an ideal top-5 that is entirely on-target | deterministic |
| precision@5 | Share of the top 5 sources that are on-target | deterministic |
| Rank quality (MRR) | Reciprocal rank of the first correct source; 1.0 = always first | deterministic |
| Expected fact in answer | Answer literally contains the required value/name/link | deterministic |
| Citation markers valid | Every `[N]` in the answer resolves to a shown source | deterministic |
| Relevance / Faithfulness / Completeness | 1–5 grades from a separate LLM judging the answer strictly against the cited sources | LLM judge |
| Speed | p50/p95 latency for retrieval and full answers | measured |

## Results (v1.4 benchmark, 2026-08-11)

| Metric | Result |
|---|---|
| Right source found | **100%** (61/61) |
| MRR | 0.94 |
| Expected fact in answer | 100% of checkable cases |
| Citation markers valid | 100% |
| Judge: relevance / faithfulness / completeness | 4.7 / 3.1 / 3.1 out of 5 |
| Median speed | ~0.5 s to sources, ~5 s to a full cited answer |

Per-category breakdown (docs/how-to, detector facts, papers, meetings &
slides, GitHub issues, services, people/roles) renders live on
`/admin/eval` — strengths and weaknesses are visible per topic, not
hidden in one blended number.

**Known weakness, stated plainly:** the judge's faithfulness score (~3/5)
shows the small generation model sometimes paraphrases loosely or picks a
less-authoritative number when sources conflict. Retrieval consistently
delivers the right material; the answer-writing model is the bottleneck.
Failing cases accumulate in a standing worklist with one-click re-run, so
candidate replacement models can be compared on exactly the hard cases.

## Why is it this good? — the configuration study

A single score says the assistant works; it does not say *which* engineering
decision made it work. The **configuration study** (`make sweep`, rendered on
`/admin/eval`) replays the whole benchmark through deliberately different
pipelines against the live index and reports each one against a control:

| Block | Configurations compared |
|---|---|
| Cumulative upgrades | the original baseline, then each upgrade added one at a time |
| top_k | 3 / 5 / 8 / 12 / 16 / 20 returned sources — quality against latency |
| Reranker | off · cross-encoder over 20 candidates · over 60 candidates |
| Embedding model | MiniLM-L6 (384-d) vs bge-m3 (1024-d), everything else identical |
| Answer model | extractive (no LLM) vs each candidate generation model |

The **baseline arm is a real reconstruction, not a memory**: the original
384-d MiniLM index is still in Qdrant, and the retriever exposes switches for
the pieces added since (rank-based fusion, acronym expansion, cross-encoder
reranking, cross-source dedup and intent shaping). Every arm answers the same
questions in the same session, so the comparison is paired.

Three practices keep the numbers honest:

- **Confidence intervals.** Every headline rate carries a 2 000-sample
  bootstrap 95% interval. On a ~60-question benchmark that band is roughly
  ±0.10 — differences smaller than it are not improvements.
- **Paired significance testing.** Each arm gets a paired bootstrap p-value
  against its block's control (the standard IR significance test; `*` = p<0.05,
  `**` = p<0.01, `***` = p<0.001). This is what stops a lucky reshuffle being
  reported as progress.
- **Retrieval arms run with generation off**, so no LLM sampling noise enters
  the retrieval numbers at all. The answer-model block is the only one that
  spends money, and there the literal "expected fact in answer" check outranks
  the LLM judge — the judge shares a model family with one of the candidates
  and could mildly favour it.

Judgments are **alternatives**: a case lists the pages a correct answer *may*
come from, so recall is undefined and precision@k is used as the coverage
measure instead.

## Reproducing

```bash
make eval LABEL=my-change      # full run with judge, ~3 min, ~$0.1
make eval-retrieval            # deterministic checks only, free, ~30 s
make sweep                     # configuration study, background, ~20 min, free
make sweep GROUPS=pipeline,top_k,rerank,embedding,generation   # + LLM block (~$0.4)
```
