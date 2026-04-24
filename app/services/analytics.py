from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, case, desc, func, select
from sqlalchemy.orm import Session

from app.models.entities import Feedback, QueryLog
from app.schemas.api import (
    AnalyticsDailyPoint,
    AnalyticsFeedbackStat,
    AnalyticsQueryCount,
    AnalyticsResponse,
    AnalyticsSummary,
    PopularQueriesResponse,
    PopularQuery,
)


def _percentile(db: Session, column, since: datetime, percentile: float) -> int | None:
    stmt = select(
        func.percentile_disc(percentile).within_group(column.asc())
    ).where(column.isnot(None), QueryLog.created_at >= since)
    value = db.scalar(stmt)
    return int(value) if value is not None else None


def build_analytics(db: Session, window_days: int = 7, limit: int = 20) -> AnalyticsResponse:
    window_days = max(1, min(window_days, 90))
    limit = max(1, min(limit, 100))
    since = datetime.now(timezone.utc) - timedelta(days=window_days)

    summary_row = db.execute(
        select(
            func.count(QueryLog.id),
            func.sum(func.cast(QueryLog.answer_generated, Integer)),
            func.sum(case((QueryLog.result_count == 0, 1), else_=0)),
            func.count(func.distinct(QueryLog.query)),
            func.avg(QueryLog.latency_ms),
            func.coalesce(func.sum(QueryLog.cost_usd), 0.0),
            func.coalesce(func.sum(QueryLog.prompt_tokens), 0),
            func.coalesce(func.sum(QueryLog.completion_tokens), 0),
        ).where(QueryLog.created_at >= since)
    ).one()

    (
        total_queries,
        answered_queries,
        zero_result_queries,
        unique_queries,
        avg_latency_ms,
        total_cost_usd,
        total_prompt_tokens,
        total_completion_tokens,
    ) = summary_row

    confidence_rows = db.execute(
        select(QueryLog.confidence, func.count(QueryLog.id))
        .where(QueryLog.created_at >= since)
        .group_by(QueryLog.confidence)
    ).all()
    confidence_breakdown = {str(key or "unknown"): count for key, count in confidence_rows}

    provider_rows = db.execute(
        select(QueryLog.generation_provider, func.count(QueryLog.id))
        .where(QueryLog.created_at >= since, QueryLog.answer_generated.is_(True))
        .group_by(QueryLog.generation_provider)
    ).all()
    provider_breakdown = {str(key or "unknown"): count for key, count in provider_rows}

    p50 = _percentile(db, QueryLog.latency_ms, since, 0.5)
    p95 = _percentile(db, QueryLog.latency_ms, since, 0.95)

    summary = AnalyticsSummary(
        window_days=window_days,
        total_queries=int(total_queries or 0),
        answered_queries=int(answered_queries or 0),
        zero_result_queries=int(zero_result_queries or 0),
        unique_queries=int(unique_queries or 0),
        avg_latency_ms=float(avg_latency_ms) if avg_latency_ms is not None else None,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_cost_usd=float(total_cost_usd or 0.0),
        total_prompt_tokens=int(total_prompt_tokens or 0),
        total_completion_tokens=int(total_completion_tokens or 0),
        confidence_breakdown=confidence_breakdown,
        provider_breakdown=provider_breakdown,
    )

    day_expr = func.date_trunc("day", QueryLog.created_at)
    daily_rows = db.execute(
        select(
            day_expr.label("day"),
            func.count(QueryLog.id),
            func.sum(func.cast(QueryLog.answer_generated, Integer)),
            func.sum(case((QueryLog.result_count == 0, 1), else_=0)),
            func.avg(QueryLog.latency_ms),
            func.percentile_disc(0.95).within_group(QueryLog.latency_ms.asc()),
            func.coalesce(func.sum(QueryLog.cost_usd), 0.0),
        )
        .where(QueryLog.created_at >= since)
        .group_by(day_expr)
        .order_by(day_expr)
    ).all()
    daily = [
        AnalyticsDailyPoint(
            day=row[0].date().isoformat() if row[0] else "",
            queries=int(row[1] or 0),
            answered=int(row[2] or 0),
            zero_result=int(row[3] or 0),
            avg_latency_ms=float(row[4]) if row[4] is not None else None,
            p95_latency_ms=int(row[5]) if row[5] is not None else None,
            total_cost_usd=float(row[6] or 0.0),
        )
        for row in daily_rows
    ]

    top_rows = db.execute(
        select(
            QueryLog.query,
            func.count(QueryLog.id),
            func.max(QueryLog.created_at),
            func.avg(QueryLog.top_score),
            func.sum(func.cast(QueryLog.answer_generated, Integer)),
        )
        .where(QueryLog.created_at >= since)
        .group_by(QueryLog.query)
        .order_by(desc(func.count(QueryLog.id)))
        .limit(limit)
    ).all()
    top_queries = [
        AnalyticsQueryCount(
            query=row[0],
            count=int(row[1]),
            last_asked_at=row[2],
            avg_top_score=float(row[3]) if row[3] is not None else None,
            answered_count=int(row[4] or 0),
        )
        for row in top_rows
    ]

    zero_rows = db.execute(
        select(
            QueryLog.query,
            func.count(QueryLog.id),
            func.max(QueryLog.created_at),
        )
        .where(QueryLog.created_at >= since, QueryLog.result_count == 0)
        .group_by(QueryLog.query)
        .order_by(desc(func.count(QueryLog.id)))
        .limit(limit)
    ).all()
    zero_result = [
        AnalyticsQueryCount(
            query=row[0],
            count=int(row[1]),
            last_asked_at=row[2],
        )
        for row in zero_rows
    ]

    low_conf_rows = db.execute(
        select(
            QueryLog.query,
            func.count(QueryLog.id),
            func.max(QueryLog.created_at),
            func.avg(QueryLog.top_score),
        )
        .where(QueryLog.created_at >= since, QueryLog.confidence == "low")
        .group_by(QueryLog.query)
        .order_by(desc(func.count(QueryLog.id)))
        .limit(limit)
    ).all()
    low_confidence = [
        AnalyticsQueryCount(
            query=row[0],
            count=int(row[1]),
            last_asked_at=row[2],
            avg_top_score=float(row[3]) if row[3] is not None else None,
        )
        for row in low_conf_rows
    ]

    feedback_rows = db.execute(
        select(Feedback.query, Feedback.rating, Feedback.comment, Feedback.created_at)
        .where(Feedback.created_at >= since, Feedback.rating.isnot(None), Feedback.rating <= 2)
        .order_by(desc(Feedback.created_at))
        .limit(limit)
    ).all()
    recent_low_rated = [
        AnalyticsFeedbackStat(
            query=row[0] or "",
            rating=int(row[1]),
            comment=row[2],
            created_at=row[3],
        )
        for row in feedback_rows
    ]

    return AnalyticsResponse(
        summary=summary,
        daily=daily,
        top_queries=top_queries,
        zero_result_queries=zero_result,
        low_confidence_queries=low_confidence,
        recent_low_rated=recent_low_rated,
    )


def build_popular_queries(
    db: Session,
    window_days: int = 7,
    limit: int = 5,
    min_results: int = 1,
    min_words: int = 3,
) -> PopularQueriesResponse:
    """Public, low-PII list of the most frequent successful queries.

    Two layers of filtering keep half-typed preview fragments out:
      - `answer_generated = true` — only count queries where the user
        actually submitted (or the speculative prefetch paired with one),
        never the retrieval-only preview fires from each keystroke.
      - `min_words` — drop 1-2 word stubs like "what", "how", "give me"
        that still sneak through via speculative pre-fires.
    Also requires at least `min_results` citations were returned, so
    failed / zero-result phrasings never surface. Query text is capped
    at 120 chars defensively.
    """
    window_days = max(1, min(window_days, 90))
    limit = max(1, min(limit, 20))
    min_words = max(1, min_words)
    since = datetime.now(timezone.utc) - timedelta(days=window_days)

    # cardinality(regexp_split_to_array(...)) gives the word count of a
    # whitespace-normalized query inside Postgres so we never tally the
    # short preview stubs.
    word_count_expr = func.cardinality(
        func.regexp_split_to_array(func.trim(QueryLog.query), r"\s+")
    )

    rows = db.execute(
        select(
            QueryLog.query,
            func.count(QueryLog.id).label("count"),
        )
        .where(
            QueryLog.created_at >= since,
            QueryLog.result_count >= min_results,
            QueryLog.scope == "public",
            QueryLog.answer_generated.is_(True),
            word_count_expr >= min_words,
        )
        .group_by(QueryLog.query)
        .order_by(desc(func.count(QueryLog.id)))
        .limit(limit)
    ).all()
    queries = [
        PopularQuery(query=(row[0] or "")[:120], count=int(row[1]))
        for row in rows
        if row[0]
    ]
    return PopularQueriesResponse(window_days=window_days, queries=queries)
