# 2025–2026 State of the Art: Comparable Systems & Proposed Improvements

*Researched 2026-08-10 (web agent). Companion to ARCHITECTURE.md — candidate roadmap items ranked by impact/effort.*

## 1. Notable comparable systems

| System | What they do better than us | What we already match/beat |
|---|---|---|
| **Archi** (CMS Ops, [arXiv:2606.04755](https://arxiv.org/abs/2606.04755)) | Ingests comms channels (Mattermost/Slack/email) + JIRA; agents query live monitoring; fully local open-weight models; MCP | We match: agentic tools, live Indico, eval panels. We beat: scholarly ingestion + dedup, SSE streaming, rate limiting |
| **MITRA** (CMS, [arXiv:2603.09800](https://arxiv.org/html/2603.09800)) | Two-tier retrieval (abstract-level first, then one document's full text); layout-preserving OCR | We beat: hybrid retrieval, 13 connectors vs one, tools, eval harness |
| **Undermind** ([undermind.ai](https://www.undermind.ai/)) | Recall-oriented iterative deep search incl. citation-graph rounds | We match: agentic search; we beat: private-corpus coverage |
| **Elicit** | Structured extraction into custom columns across hundreds of papers | We beat: non-paper sources, meetings, people |
| **Scite** | Claim-level citation stance (supporting/contrasting) over 1.2B statements | We beat: everything except citation-stance |
| **Glean** ([why custom embeddings](https://jxnl.co/writing/2025/09/11/why-glean-builds-custom-embedding-models-for-every-customer/)) | Self-learning ranking from query-click pairs (~20% gain/6mo); org-graph personalization | We match: hybrid + rerank + permissions; we beat: physics-source awareness, cost |
| **Onyx** (ex-Danswer) | 40+ connectors, air-gapped 37K-user deployments, usage dashboards | We beat: retrieval sophistication, eval harness |
| **Perplexity Enterprise** | Blended web + org answers; 400+ connectors incl. MCP | We beat: domain retrieval depth, dedup, eval, cost |

## 2. Top 8 improvements (impact/effort ranked)

1. **Contextual retrieval at ingestion** ([Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)) — prepend cheap-LLM chunk context before embedding AND FTS indexing; 49-67% fewer retrieval failures; one-time cost per chunk (content-hash idempotency). Effort: days + reindex.
2. **Embedding upgrade** MiniLM-384 → Arctic Embed 2.0 / BGE-M3 / Qwen3-Embedding-0.6B ([comparison](https://milvus.io/blog/choose-embedding-model-rag-2026.md)) — largest single recall gain; provider already pluggable. Effort: config + reindex.
3. **Close the feedback loop** (Glean pattern) — per-document boosts from query_logs + feedback ratings; later fine-tune cross-encoder on (query, clicked-chunk) pairs. Effort: days → weeks.
4. **Corrective self-check gate** — reuse the eval judge inline on low-confidence answers; on failed groundedness, re-retrieve once with refined query before streaming a weak answer. Effort: days.
5. **Two-tier document-first retrieval** (MITRA) — doc-level pass then chunk search scoped to winning docs; fixes chunk-bleed across TDR-length PDFs; generalizes existing parent-expansion. Effort: days.
6. **Supersedes-chain temporal layer** — explicit version edges (GitHub releases, Zenodo version DOIs, arXiv v2+); hard-demote superseded versions; extends existing DOI/arXiv dedup. Effort: days.
7. **Mattermost/ticket connector** (Archi's edge) — where "why does reco crash" answers actually live; channel visibility → existing public/internal model. Effort: weeks.
8. **LazyGraphRAG global-question mode** ([Microsoft](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)) — entity/community graph (extend people-graph extraction to subsystems/software), query-time map-reduce for "summarize status of X" questions. Effort: weeks.

## 3. Sleeper: "answer-or-ask" expert routing

When the groundedness gate fails, don't apologize — route to a human. We already have the people graph (papers + meetings + ORCID): "Low-confidence answer. This topic was last presented by J. Smith at the DAQ WG meeting on 2026-07-14 [link]; 6 related documents." No comparable system does expert referral on retrieval failure. Effort: days — a join between the failing path and the people graph.
