import pytest
from pydantic import ValidationError

from app.schemas.api import FeedbackCreate, FeedbackUpdate, QueryRequest


def test_nul_stripped_from_query_and_history() -> None:
    req = QueryRequest(
        query="foo\x00bar",
        history=[{"role": "user", "content": "a\x00b"}],
    )
    assert req.query == "foobar"
    assert req.history[0].content == "ab"


def test_all_nul_query_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(query="\x00\x00")
    with pytest.raises(ValidationError):
        QueryRequest(query="ok", history=[{"role": "user", "content": "\x00"}])


def test_nul_stripped_from_feedback() -> None:
    fb = FeedbackCreate(query="q\x00", comment=" bad\x00 answer ")
    assert (fb.query, fb.comment) == ("q", "bad answer")
    assert FeedbackUpdate(comment="x\x00y").comment == "xy"
    with pytest.raises(ValidationError):
        FeedbackUpdate(comment="\x00 ")
    with pytest.raises(ValidationError):
        FeedbackCreate(comment="\x00")
