"""build_popular_queries: public chips are window-clamped and need count >= 2."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.dialects import postgresql

from app.services import analytics


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeDb:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.statements = []

    def execute(self, stmt):
        self.statements.append(stmt)
        return _Result(self.rows)


def _compiled(stmt):
    return stmt.compile(dialect=postgresql.dialect())


@pytest.mark.parametrize(
    ("requested", "effective"), [(3650, 90), (91, 90), (90, 90), (7, 7), (0, 1), (-5, 1)]
)
def test_window_days_clamped(requested: int, effective: int) -> None:
    db = FakeDb()
    before = datetime.now(timezone.utc)
    resp = analytics.build_popular_queries(db, window_days=requested)
    assert resp.window_days == effective
    params = _compiled(db.statements[0]).params
    since = [v for v in params.values() if isinstance(v, datetime)]
    assert len(since) == 1
    expected = before - timedelta(days=effective)
    assert abs((since[0] - expected).total_seconds()) < 60


def test_requires_min_count_of_two() -> None:
    db = FakeDb(rows=[("how do I run the simulation", 4)])
    resp = analytics.build_popular_queries(db, window_days=7)
    compiled = _compiled(db.statements[0])
    sql = str(compiled)
    assert "HAVING count(query_logs.id) >=" in sql
    assert analytics.POPULAR_MIN_COUNT == 2
    assert analytics.POPULAR_MIN_COUNT in compiled.params.values()
    assert [q.query for q in resp.queries] == ["how do I run the simulation"]
