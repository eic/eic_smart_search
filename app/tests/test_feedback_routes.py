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
