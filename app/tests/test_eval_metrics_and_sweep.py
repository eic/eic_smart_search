"""Retrieval metrics, uncertainty, and the configuration-sweep arm matrix."""
from __future__ import annotations

import pytest

from app.eval.metrics import mean_ci, paired_bootstrap_p, rank_metrics, stars
from app.eval.sweep import DEFAULT_GROUPS, GROUPS, Arm, _case_scores


def test_rank_metrics_perfect_and_missed() -> None:
    # Expected URLs are alternatives: any of them makes a citation relevant.
    m = rank_metrics(["https://a/x"] * 3 + ["https://b/y"] * 2, ["a/x", "b/y"])
    assert m["hit"] is True and m["rank"] == 1
    assert m["ndcg@5"] == pytest.approx(1.0) and m["precision@5"] == 1.0 and m["success@1"] == 1.0

    missed = rank_metrics(["https://c/z"], ["nope"])
    assert missed["hit"] is False and missed["ndcg@5"] == 0.0 and missed["precision@5"] == 0.0


def test_ndcg_rewards_higher_rank_and_never_exceeds_one() -> None:
    early = rank_metrics(["https://a/x", "https://q/1", "https://q/2"], ["a/x"])
    late = rank_metrics(["https://q/1", "https://q/2", "https://a/x"], ["a/x"])
    assert early["ndcg@5"] > late["ndcg@5"]
    # Several chunks of the same expected page must not push nDCG above 1.
    duplicated = rank_metrics(["https://a/x#1", "https://a/x#2", "https://a/x#3"], ["a/x"])
    assert duplicated["ndcg@3"] == pytest.approx(1.0)
    assert duplicated["ndcg@10"] < 1.0


def test_precision_penalizes_off_target_padding() -> None:
    focused = rank_metrics(["https://a/x", "https://a/x2"], ["a/x"])
    padded = rank_metrics(["https://a/x", "https://q/1", "https://q/2", "https://q/3"], ["a/x"])
    # Denominator is always k, so returning fewer sources is never a free win.
    assert focused["precision@5"] == 0.4 and padded["precision@5"] == 0.2
    assert focused["ndcg@5"] > padded["ndcg@5"]
    assert rank_metrics(["https://a/x"], ["a/x"])["ndcg@1"] == 1.0


def test_no_expected_url_yields_no_metrics() -> None:
    assert rank_metrics(["https://a"], []) == {}


def test_mean_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    first = mean_ci(values)
    assert first == mean_ci(values)  # fixed seed → reproducible dashboards
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert mean_ci([]) is None


def test_paired_bootstrap_detects_real_and_ignores_noise() -> None:
    baseline = [0.0] * 20
    much_better = [1.0] * 20
    assert paired_bootstrap_p(baseline, much_better) < 0.05
    identical = paired_bootstrap_p(baseline, list(baseline))
    assert identical == 1.0
    assert stars(0.0005) == "***" and stars(0.2) == ""


def test_sweep_groups_cover_defaults_and_build_arms() -> None:
    for group in DEFAULT_GROUPS:
        arms = GROUPS[group]()
        assert arms and all(isinstance(a, Arm) for a in arms)
        assert len({a.key for a in arms}) == len(arms)
    pipeline = GROUPS["pipeline"]()
    # The first arm must be the reconstructed original system, the last the
    # current production configuration (no overrides at all).
    assert pipeline[0].settings["EMBEDDING_MODEL"].endswith("all-MiniLM-L6-v2")
    assert pipeline[0].retriever["fusion"] == "weighted"
    assert pipeline[-1].settings == {} and pipeline[-1].retriever == {}
    assert all(a.generate for a in GROUPS["generation"]())


def test_generation_arms_add_alt_providers_when_keys_set() -> None:
    class S:
        GEMINI_API_KEY = "g-key"
        GEMINI_MODEL = "gemini-2.5-flash"
        DEEPSEEK_API_KEY = None
        GROQ_API_KEY = None
        OPENROUTER_API_KEY = ""

    from app.eval.sweep import generation_arms

    keys = {a.key for a in generation_arms(S())}
    assert "gen_gemini-2.5-flash" in keys          # key set → arm exists
    assert not any("deepseek" in k for k in keys)  # no key → no arm
    gem = next(a for a in generation_arms(S()) if a.key == "gen_gemini-2.5-flash")
    assert gem.settings["OPENAI_BASE_URL"].startswith("https://generativelanguage")
    assert gem.settings["OPENAI_API_KEY"] == "g-key"
    # No settings at all → only the OpenAI + extractive arms.
    assert {a.key for a in generation_arms(None)} == {
        "gen_extractive", "gen_gpt-5.4-nano", "gen_gpt-5.6-luna",
    }


def test_case_scores_align_and_default_to_zero() -> None:
    results = [{"retrieval": {"ndcg@5": 0.5}}, {"error": "boom"}, {"retrieval": {}}]
    assert _case_scores(results, "ndcg@5") == [0.5, 0.0, 0.0]


def test_nrp_arms_carry_thinking_switches_and_list_arms_filters() -> None:
    class S:
        ANTHROPIC_API_KEY = GEMINI_API_KEY = DEEPSEEK_API_KEY = GROQ_API_KEY = OPENROUTER_API_KEY = None
        NRP_API_KEY = "nrp-key"
        NRP_MODEL = "qwen3, gpt-oss,glm-5,kimi"

    from app.eval.sweep import generation_arms, list_arms

    arms = {a.key: a for a in generation_arms(S())}
    assert arms["gen_qwen3"].settings["OPENAI_BASE_URL"] == "https://ellm.nrp-nautilus.io/v1"
    assert arms["gen_qwen3"].settings["OPENAI_EXTRA_BODY"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert arms["gen_qwen3"].settings["OPENAI_REQUEST_TIMEOUT_S"] == 120.0
    assert arms["gen_gpt-oss"].settings["OPENAI_REASONING_EFFORT"] == "low"
    assert arms["gen_glm-5"].settings["OPENAI_MAX_OUTPUT_TOKENS"] == 1500
    assert "OPENAI_EXTRA_BODY" not in arms["gen_kimi"].settings
    assert arms["gen_kimi"].settings["OPENAI_MAX_OUTPUT_TOKENS"] == 2500

    chosen = list_arms(S(), ["generation"], ["gen_gpt-5.4-nano", "gen_qwen3"])
    assert [a.key for a in chosen] == ["gen_gpt-5.4-nano", "gen_qwen3"]
    with pytest.raises(ValueError, match="unknown sweep arms"):
        list_arms(S(), ["generation"], ["gen_nope"])


@pytest.mark.parametrize("warm", [True, False])
def test_build_answer_generator_warm_indico_switch(warm: bool) -> None:
    """Sweep arms build one generator each; warm_indico=False keeps them from
    each starting an Indico prefetch thread. Production default warms."""
    from unittest.mock import patch

    from app.core.config import get_settings
    from app.integrations.indico import IndicoClient
    from app.llm.generation import build_answer_generator

    settings = get_settings().model_copy(update={
        "GENERATION_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "INDICO_ENABLED": True,
        "INDICO_CATEGORY_URL": "https://indico.bnl.gov/category/402/",
        "INDICO_CACHE_TTL_S": 300.0,
    })
    spawned: list = []
    with patch.object(IndicoClient, "_spawn", lambda self, target, *args: spawned.append(target)):
        kwargs = {} if warm else {"warm_indico": False}
        generator = build_answer_generator(settings, **kwargs)
    assert generator._indico is not None
    assert len(spawned) == (1 if warm else 0)
