# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""faithgap — a bring-your-own-model faithfulness audit for LLM recommenders.

Measures the *faithfulness gap*: how much more a model's recommendation moves when
you change the MEANING of its stated reasoning (flip / swap / erase) versus when you
only reword it (paraphrase control). A large positive gap means the pick actually
tracks the stated reason; a gap near zero means the reasoning is decorative.

Quickstart (offline, no keys)::

    from faithgap import FaithfulnessAudit, Case, MockModel

    cases = [Case(history=["Halo", "Doom"],
                  candidates=["Call of Duty", "Stardew Valley", "Tetris"],
                  positive="Call of Duty")]
    result = FaithfulnessAudit(model=MockModel()).run(cases)
    print(result.summary())

Method and full results: Shah, "How Faithful Is the Reasoning of LLM Recommenders?
A Counterfactual Audit," RecSys 2026 (doi:10.1145/3773078.3841294).
"""
from __future__ import annotations

from .audit import CaseRecord, EditOutcome, FaithfulnessAudit, Result
from .cases import Case
from .model import MockModel, as_model
from .prompts import (
    CONDITIONED_SYSTEM, REASON_SYSTEM,
    conditioned_prompt, extract_reason, extract_think, parse_ranking, reason_prompt,
)

__version__ = "0.1.0"

__all__ = [
    "FaithfulnessAudit",
    "Result",
    "Case",
    "MockModel",
    "as_model",
    "CaseRecord",
    "EditOutcome",
    "reason_prompt",
    "conditioned_prompt",
    "parse_ranking",
    "extract_reason",
    "extract_think",
    "REASON_SYSTEM",
    "CONDITIONED_SYSTEM",
]
