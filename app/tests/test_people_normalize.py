"""Unit tests for the people-graph normalization helpers.

These don't need a database; they exercise the pure parsing/dedup logic
that the builder relies on to collapse "Müller, Hans" and "Hans Muller"
into the same row.
"""
from app.people.normalize import (
    join_indico_person,
    merge_unique,
    normalize_name,
    parse_author_string,
)


def test_normalize_strips_diacritics_and_lowercases():
    assert normalize_name("Müller") == "muller"
    # Token-sorted: "Côté" sorts before "Élise" alphabetically.
    assert normalize_name("Élise Côté") == "cote elise"


def test_normalize_collapses_punctuation_and_whitespace():
    # Token-sorted: alphabetical order, regardless of the input order.
    assert normalize_name("  J.P. Morgan ") == "j morgan p"
    assert normalize_name("Anne-Marie\tO'Brien") == "anne-marie o'brien"


def test_normalize_is_order_invariant_across_last_first_and_first_last():
    # The whole point of token-sort: "Last, First" and "First Last"
    # — same person, different conference template — collapse together.
    assert normalize_name("Diefenthaler, Markus") == normalize_name("Markus Diefenthaler")


def test_normalize_returns_empty_for_garbage():
    assert normalize_name("") == ""
    assert normalize_name("   ") == ""
    assert normalize_name("---") == ""


def test_parse_zenodo_last_first_with_affiliation():
    parsed = parse_author_string("Müller, Hans (Brookhaven National Lab)")
    assert parsed is not None
    assert parsed.full_name == "Hans Müller"
    assert parsed.affiliation == "Brookhaven National Lab"


def test_parse_zenodo_first_last_no_affiliation():
    parsed = parse_author_string("Hans Müller")
    assert parsed is not None
    assert parsed.full_name == "Hans Müller"
    assert parsed.affiliation is None


def test_parse_zenodo_handles_blank_first_after_comma():
    # Some records ship as just "Last," — treat as a single token.
    parsed = parse_author_string("Smith,")
    assert parsed is not None
    assert parsed.full_name == "Smith"


def test_parse_zenodo_returns_none_on_empty():
    assert parse_author_string("") is None
    assert parse_author_string("   ") is None


def test_join_indico_person_prefers_full_name():
    parsed = join_indico_person(
        {"fullName": "Hans Müller", "first_name": "Hans", "last_name": "Mueller", "affiliation": "BNL"}
    )
    assert parsed is not None
    assert parsed.full_name == "Hans Müller"
    assert parsed.affiliation == "BNL"


def test_join_indico_person_falls_back_to_first_last():
    parsed = join_indico_person({"first_name": "Hans", "last_name": "Mueller"})
    assert parsed is not None
    assert parsed.full_name == "Hans Mueller"


def test_join_indico_person_returns_none_when_no_name():
    assert join_indico_person({"affiliation": "BNL"}) is None
    assert join_indico_person({}) is None
    assert join_indico_person(None) is None  # type: ignore[arg-type]


def test_merge_unique_preserves_order_and_skips_case_dups():
    out = merge_unique(["Brookhaven"], ["BNL", "brookhaven", "Jefferson Lab", " "])
    assert out == ["Brookhaven", "BNL", "Jefferson Lab"]


def test_normalize_collapses_zenodo_and_indico_to_same_key():
    # The whole point: a Zenodo "Last, First (Affil)" string and an Indico
    # `fullName` for the *same* person must hash to the same normalized key
    # so the builder dedupes. We only commit to diacritic-folding (ü→u),
    # not ASCII transliteration (ü→ue) — those are different spellings
    # and the matcher leaves them as separate rows on purpose.
    zen = parse_author_string("Müller, Hans (BNL)")
    ind = join_indico_person({"fullName": "Hans Müller"})
    assert zen and ind
    assert normalize_name(zen.full_name) == normalize_name(ind.full_name) == "hans muller"
