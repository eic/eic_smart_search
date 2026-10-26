from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import create_app
from app.models.entities import CitationClick, Feedback


class FakeDb:
    """Records added rows and fills column defaults on commit, as a flush would."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        for obj in self.added:
            obj.id = obj.id or "row-1"
            obj.created_at = obj.created_at or datetime.now(UTC)


@pytest.fixture
def client_and_db():
    app = create_app()
    db = FakeDb()
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app), db


def test_citation_click_stores_row(client_and_db) -> None:
    client, db = client_and_db
    resp = client.post("/citations/click", json={"query_log_id": "q1", "chunk_id": "c1"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"id", "created_at"}
    (row,) = db.added
    assert isinstance(row, CitationClick)
    assert (row.query_log_id, row.chunk_id) == ("q1", "c1")


def test_feedback_without_rating_or_comment_rejected(client_and_db) -> None:
    client, db = client_and_db
    resp = client.post(
        "/feedback",
        json={"query_log_id": "q1", "selected_citation_ids": ["c1"], "metadata": {"type": "citation_click"}},
    )
    assert resp.status_code == 422
    assert db.added == []


def test_feedback_rating_only_ok(client_and_db) -> None:
    client, db = client_and_db
    resp = client.post("/feedback", json={"query_log_id": "q1", "rating": 5})
    assert resp.status_code == 200
    (row,) = db.added
    assert isinstance(row, Feedback) and row.rating == 5


class IntegrityFailDb(FakeDb):
    """Commit fails like a query_log_id FK violation would."""

    def __init__(self) -> None:
        super().__init__()
        self.rolled_back = False

    def commit(self) -> None:
        from sqlalchemy.exc import IntegrityError

        raise IntegrityError("INSERT ...", {}, Exception("violates foreign key constraint"))

    def rollback(self) -> None:
        self.rolled_back = True


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/citations/click", {"query_log_id": "nope", "chunk_id": "c1"}),
        ("/feedback", {"query_log_id": "nope", "rating": 4}),
    ],
)
def test_unknown_query_log_id_is_404(path, body) -> None:
    app = create_app()
    db = IntegrityFailDb()
    app.dependency_overrides[get_db] = lambda: db
    resp = TestClient(app).post(path, json=body)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "unknown query_log_id"
    assert db.rolled_back


def test_feedback_whitespace_comment_without_rating_rejected(client_and_db) -> None:
    client, db = client_and_db
    resp = client.post("/feedback", json={"query_log_id": "q1", "comment": "   \n "})
    assert resp.status_code == 422
    assert db.added == []


def test_feedback_comment_stored_stripped(client_and_db) -> None:
    client, db = client_and_db
    resp = client.post("/feedback", json={"rating": 2, "comment": "  wrong answer \n"})
    assert resp.status_code == 200
    (row,) = db.added
    assert row.comment == "wrong answer"


def test_feedback_patch_strips_and_rejects_blank(client_and_db) -> None:
    client, db = client_and_db
    row = Feedback(rating=1)
    row.id, row.created_at = "fb-1", datetime.now(UTC)
    db.get = lambda model, key, **kw: row if key == "fb-1" else None
    assert client.patch("/feedback/fb-1", json={"comment": "   "}).status_code == 422
    resp = client.patch("/feedback/fb-1", json={"comment": "  stale link  "})
    assert resp.status_code == 200
    assert row.comment == "stale link"


def test_query_with_nul_byte_not_500() -> None:
    """Postgres rejects NUL in text columns; the schema strips it so the
    QueryLog insert can't 500. An all-NUL query is a 422."""
    from app.api.routes import query_service_dep

    app = create_app()
    app.dependency_overrides[query_service_dep] = lambda: None
    assert TestClient(app).post("/query", json={"query": "\x00"}).status_code == 422


def _patchable(db, row) -> None:
    db.get = lambda model, key, **kw: row if key == row.id else None


def test_feedback_patch_rejects_when_comment_already_set(client_and_db) -> None:
    client, db = client_and_db
    row = Feedback(rating=1, comment="first")
    row.id, row.created_at = "fb-2", datetime.now(UTC)
    _patchable(db, row)
    resp = client.patch("/feedback/fb-2", json={"comment": "overwrite"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "feedback comment already set"
    assert row.comment == "first"


def test_feedback_patch_rejects_old_rows_and_unknown_ids(client_and_db) -> None:
    from datetime import timedelta

    client, db = client_and_db
    row = Feedback(rating=1)
    row.id, row.created_at = "fb-3", datetime.now(UTC) - timedelta(minutes=16)
    _patchable(db, row)
    resp = client.patch("/feedback/fb-3", json={"comment": "late"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "feedback too old to edit"
    assert row.comment is None
    assert client.patch("/feedback/missing", json={"comment": "x"}).status_code == 404
    # Naive timestamps are treated as UTC; a fresh one is accepted once.
    row.created_at = datetime.now(UTC).replace(tzinfo=None)
    assert client.patch("/feedback/fb-3", json={"comment": "ok"}).status_code == 200
    assert client.patch("/feedback/fb-3", json={"comment": "again"}).status_code == 409


def test_feedback_query_length_capped(client_and_db) -> None:
    client, db = client_and_db
    resp = client.post("/feedback", json={"rating": 3, "query": "q" * 2001})
    assert resp.status_code == 422
    assert db.added == []
