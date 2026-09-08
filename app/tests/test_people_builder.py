"""Unit tests for the people-graph builder helpers.

Covers the two non-trivial bits:
    1. Convener-line regex + role-prefix stripping (handles the
       "Conveners Sasha Prozorov ( … )" shape from epic-eic.org/sc).
    2. Email-based merge across two accumulators in one batch.
       The DB-side merge path (merge against an existing row) is exercised
       via integration runs, not here.
"""
from app.people.builder import (
    _NAME_EMAIL_RE,
    _PersonAccumulator,
    _email_local_fingerprint,
    _merge_by_email,
    _strip_role_prefix,
)
from app.people.normalize import normalize_name


def test_name_email_regex_matches_simple_pair():
    text = "- Sasha Prozorov ( alexandr.prozorov@cvut.cz )"
    matches = list(_NAME_EMAIL_RE.finditer(text))
    assert len(matches) == 1
    assert matches[0].group(1) == "Sasha Prozorov"
    assert matches[0].group(2) == "alexandr.prozorov@cvut.cz"


def test_name_email_regex_finds_multiple_in_one_chunk():
    text = (
        "- Conveners Sasha Prozorov ( alexandr.prozorov@cvut.cz ) "
        "Stephen Kay ( stephen.kay@york.ac.uk )"
    )
    matches = list(_NAME_EMAIL_RE.finditer(text))
    assert len(matches) == 2
    assert matches[0].group(2) == "alexandr.prozorov@cvut.cz"
    assert matches[1].group(2) == "stephen.kay@york.ac.uk"


def test_name_email_regex_skips_lowercase_names():
    # We don't want "and the project ( foo@x.org )" to count.
    text = "the working group ( foo@x.org )"
    assert list(_NAME_EMAIL_RE.finditer(text)) == []


def test_strip_role_prefix_drops_leading_label():
    assert _strip_role_prefix("Conveners Sasha Prozorov") == "Sasha Prozorov"
    assert _strip_role_prefix("Co-Chair Stephen Kay") == "Stephen Kay"
    assert _strip_role_prefix("Members Hans Müller") == "Hans Müller"


def test_strip_role_prefix_leaves_clean_names_alone():
    assert _strip_role_prefix("Sasha Prozorov") == "Sasha Prozorov"
    assert _strip_role_prefix("Markus Diefenthaler") == "Markus Diefenthaler"


def test_merge_by_email_collapses_two_in_batch_accumulators():
    # Two name forms for the same person, joined only by email. Without the
    # email-merge pass the flush would write two rows.
    sasha = _PersonAccumulator(
        normalized=normalize_name("Sasha Prozorov"), canonical_name="Sasha Prozorov"
    )
    sasha.add_email("alexandr.prozorov@cvut.cz")
    sasha.add_working_group("User Learning Working Group")
    sasha.add_activity({"kind": "working_group_role", "title": "User Learning"})

    formal = _PersonAccumulator(
        normalized=normalize_name("Prozorov, Alexandr"),
        canonical_name="Prozorov, Alexandr",
    )
    formal.add_email("alexandr.prozorov@cvut.cz")
    formal.add_activity({"kind": "talk", "title": "Onboarding resources"})
    formal.add_activity({"kind": "talk", "title": "Documentation and discoverability"})

    upserts = {sasha.normalized: sasha, formal.normalized: formal}

    # No DB existing rows — pass a stub session that returns empty.
    class _StubResult:
        @staticmethod
        def scalars():
            class _S:
                @staticmethod
                def all():
                    return []
            return _S()

    class _StubDb:
        @staticmethod
        def execute(_stmt):
            return _StubResult()

    merges = _merge_by_email(_StubDb(), upserts)
    assert merges == 1
    # The accumulator with more activity entries (formal, 2 entries) keeps
    # the row; the website-only one is folded in as an alias.
    assert len(upserts) == 1
    survivor = next(iter(upserts.values()))
    assert "Sasha Prozorov" in survivor.aliases
    assert "User Learning Working Group" in survivor.working_groups
    # Activity from both accumulators is merged.
    titles = [a.get("title") for a in survivor.activity]
    assert "User Learning" in titles
    assert "Onboarding resources" in titles


def test_merge_by_email_collapses_via_local_part_fingerprint():
    """The most important case: Indico accumulator has no email, website
    accumulator does. The fingerprint of "alexandr.prozorov@cvut.cz"
    collides with the Indico "Prozorov, Alexandr" normalized name.
    """
    sasha_web = _PersonAccumulator(
        normalized=normalize_name("Sasha Prozorov"), canonical_name="Sasha Prozorov"
    )
    sasha_web.add_email("alexandr.prozorov@cvut.cz")
    sasha_web.add_working_group("User Learning Working Group")
    sasha_web.add_activity({"kind": "working_group_role", "title": "User Learning"})

    formal_indico = _PersonAccumulator(
        normalized=normalize_name("Prozorov, Alexandr"),  # "alexandr prozorov"
        canonical_name="Prozorov, Alexandr",
    )
    # Note: NO email set — that's the realistic Indico path.
    formal_indico.add_activity({"kind": "talk", "title": "Onboarding resources"})

    upserts = {sasha_web.normalized: sasha_web, formal_indico.normalized: formal_indico}

    class _StubResult:
        @staticmethod
        def scalars():
            class _S:
                @staticmethod
                def all():
                    return []
            return _S()

    class _StubDb:
        @staticmethod
        def execute(_stmt):
            return _StubResult()

    merges = _merge_by_email(_StubDb(), upserts)
    assert merges == 1
    assert len(upserts) == 1
    keeper = next(iter(upserts.values()))
    # Formal Indico key wins because it matches the email's local fingerprint.
    assert keeper.normalized == "alexandr prozorov"
    assert "Sasha Prozorov" in keeper.aliases
    assert keeper.email == "alexandr.prozorov@cvut.cz"
    assert "User Learning Working Group" in keeper.working_groups


def test_email_local_fingerprint_handles_separators():
    assert _email_local_fingerprint("alexandr.prozorov@x.cz") == "alexandr prozorov"
    assert _email_local_fingerprint("hans-peter_mueller@bnl.gov") == "hans mueller peter"


def test_email_local_fingerprint_skips_single_token_locals():
    # Generic / institutional addresses must not act as a merge signal.
    assert _email_local_fingerprint("info@x.cz") == ""
    assert _email_local_fingerprint("asmith@x.cz") == ""


def test_merge_by_email_is_no_op_when_no_emails():
    a = _PersonAccumulator("a foo", "A Foo")
    b = _PersonAccumulator("b bar", "B Bar")
    upserts = {a.normalized: a, b.normalized: b}

    class _StubDb:
        @staticmethod
        def execute(_stmt):  # never called when there are no emails
            raise AssertionError("execute should not run with empty email set")

    merges = _merge_by_email(_StubDb(), upserts)
    assert merges == 0
    assert len(upserts) == 2


def test_accumulator_orcid_first_wins_and_absorbs():
    from app.people.builder import _PersonAccumulator

    acc = _PersonAccumulator(normalized="doe jane", canonical_name="Jane Doe")
    acc.add_orcid("0000-0001-2345-6789")
    acc.add_orcid("0000-9999-9999-9999")  # later sighting must not overwrite
    assert acc.orcid == "0000-0001-2345-6789"

    other = _PersonAccumulator(normalized="doe jane", canonical_name="Jane Doe")
    other.add_orcid("0000-0002-0000-0000")
    blank = _PersonAccumulator(normalized="doe jane", canonical_name="Jane Doe")
    blank.absorb(other)
    assert blank.orcid == "0000-0002-0000-0000"
