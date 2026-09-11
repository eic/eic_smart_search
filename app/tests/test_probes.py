from app.services.probes import classify_probe


def test_cli_subcommand_guesses_are_probes() -> None:
    for word in ("init", "start", "burn-snapshot", "eic_ask eic-ask"):
        assert classify_probe(word, "cli") == "short-cli"
    assert classify_probe("how do I update eic-shell", "cli") is None


def test_short_widget_queries_are_not_probes() -> None:
    assert classify_probe("eic-shell", None) is None
    assert classify_probe("EICrecon", None) is None


def test_graphql_and_symbol_probes() -> None:
    assert classify_probe("queryy { __typename }", None) == "graphql"
    assert classify_probe("query @skip {__typename}", None) == "graphql"
    assert classify_probe("query ! {__typename}", None) == "graphql"
    assert classify_probe("{{{", None) == "no-letters"
    assert classify_probe("what is a query language?", None) is None
