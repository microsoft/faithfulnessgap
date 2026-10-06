# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The one input type a researcher builds: a ``Case``.

A case is a single audit instance — a user's history, the candidate pool to rank,
and the held-out ground-truth next item. You bring a list of these (from your own
dataset, any domain), hand them to ``FaithfulnessAudit.run(...)``, and get the gap.

Items may be plain strings (titles) or ids paired with a ``titles`` mapping on the
audit. Keep it simple: strings are fine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


@dataclass
class Case:
    """One user to audit.

    Attributes
    ----------
    history   : the user's past items, oldest to newest (context for the model).
    candidates: the pool to rank, including the ground-truth ``positive``.
    positive  : the held-out next item (must appear in ``candidates``).
    user      : optional id, only used for labeling/output.
    """
    history: Sequence[Any]
    candidates: Sequence[Any]
    positive: Any
    user: Optional[Any] = None

    def __post_init__(self):
        self.history = list(self.history)
        self.candidates = list(self.candidates)
        if self.positive not in self.candidates:
            raise ValueError(
                f"positive {self.positive!r} is not in candidates for user {self.user!r}; "
                "the ground-truth next item must be one of the ranked candidates."
            )
