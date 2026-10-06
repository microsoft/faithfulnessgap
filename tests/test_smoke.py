# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Smoke tests — offline, no network, deterministic. Run with: pytest -q"""
from faithgap import Case, FaithfulnessAudit, MockModel
from faithgap.edits import heuristic_flip, heuristic_paraphrase
from faithgap.prompts import parse_ranking


def _cases():
    return [
        Case(history=["Halo shooter", "Doom shooter", "Call of Duty shooter"],
             candidates=["Battlefield shooter", "Stardew farming", "Tetris puzzle"],
             positive="Battlefield shooter", user="a"),
        Case(history=["Stardew farming", "Harvest Moon farming", "Animal Crossing cozy"],
             candidates=["Story of Seasons farming", "Doom shooter", "Forza racing"],
             positive="Story of Seasons farming", user="b"),
        Case(history=["Forza racing", "Gran Turismo racing", "Need for Speed racing"],
             candidates=["F1 racing", "Tetris puzzle", "Animal Crossing cozy"],
             positive="F1 racing", user="c"),
    ]


def test_case_rejects_positive_not_in_candidates():
    import pytest
    with pytest.raises(ValueError):
        Case(history=["x"], candidates=["a", "b"], positive="z")


def test_parse_ranking_is_full_permutation():
    labels = ["a", "b", "c"]
    # only mentions 3 and 1; 2 must be appended -> full permutation, no dups
    out = parse_ranking("RANKING: 3, 1", labels)
    assert sorted(out) == sorted(labels)
    assert out[0] == "c" and out[1] == "a"


def test_heuristic_edits():
    assert heuristic_flip("The user likes shooters") != "The user likes shooters"
    assert heuristic_paraphrase("The user likes shooters").lower().startswith("to summarize")
    # flip with no preference words still changes the text (prefix fallback)
    assert heuristic_flip("abc def") != "abc def"


def test_audit_runs_and_is_deterministic():
    audit = FaithfulnessAudit(model=MockModel(), seed=7)
    r1 = audit.run(_cases(), bootstrap=200)
    r2 = audit.run(_cases(), bootstrap=200)
    assert r1.n == 3
    assert r1.faithfulness_gap == r2.faithfulness_gap          # deterministic
    assert r1.ci == r2.ci
    assert set(r1.per_op) == {"flip", "swap", "erase", "paraphrase"}
    # rates are valid probabilities
    for v in r1.per_op.values():
        assert 0.0 <= v <= 1.0
    assert -1.0 <= r1.faithfulness_gap <= 1.0


def test_mockmodel_is_meaning_sensitive():
    # On MockModel the pick tracks the reasoning, so the gap should be clearly positive.
    audit = FaithfulnessAudit(model=MockModel())
    r = audit.run(_cases(), bootstrap=0)
    assert r.faithfulness_gap > 0.0
    assert r.ci is None   # bootstrap=0 skips the CI


def test_to_dict_shape():
    r = FaithfulnessAudit(model=MockModel()).run(_cases(), bootstrap=100)
    d = r.to_dict()
    assert set(d) >= {"n", "faithfulness_gap", "meaning_change_rate",
                      "paraphrase_change_rate", "per_op_change_rate", "verdict"}
    assert d["verdict"] in ("faithful", "mixed", "decorative")


def test_custom_decision_readout():
    # top-1 is the default; a custom decision (rank of the held-out item) also runs
    # and stays a valid gap. Both one- and two-arg decisions are accepted.
    top3 = FaithfulnessAudit(model=MockModel(), decision=lambda rank: tuple(rank[:3]))
    pos_rank = FaithfulnessAudit(
        model=MockModel(), decision=lambda rank, case: rank.index(case.positive))
    for audit in (top3, pos_rank):
        r = audit.run(_cases(), bootstrap=0)
        assert -1.0 <= r.faithfulness_gap <= 1.0
        assert set(r.per_op) == {"flip", "swap", "erase", "paraphrase"}
