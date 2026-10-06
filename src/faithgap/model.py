# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The model seam: faithgap talks to any model through one tiny contract.

A *model* is any callable ``model(prompt, system=None) -> str``. Wrap your own
LLM (an API, a local checkpoint, vLLM, anything) in such a callable and the whole
audit runs against it unchanged. No base class to inherit; a plain function works.

``MockModel`` is a dependency-free, offline stand-in used by the quickstart and
tests. It is *meaning-sensitive* on purpose: it ranks candidates by keyword overlap
with the supplied reasoning and honors its stated polarity (a negated analysis inverts
the ranking), so editing the reasoning visibly moves — or, for a reword, doesn't move —
its pick. Enough to demonstrate the audit end-to-end with no network and no keys.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

# A model is just this. Type alias kept loose so a bare lambda satisfies it.
Model = Callable[..., str]


def as_model(fn: Model) -> Model:
    """Normalize a user callable to the ``(prompt, system=None)`` calling convention.

    Accepts callables that take only ``prompt`` as well as ones that accept a
    ``system`` keyword, so researchers can pass whichever they already have.
    """
    def _call(prompt: str, system: Optional[str] = None) -> str:
        try:
            return fn(prompt, system=system)
        except TypeError:
            return fn(prompt)
    return _call


_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "is",
    "are", "they", "their", "this", "that", "these", "those", "user", "users",
    "history", "shows", "suggests", "preference", "preferences", "items", "item",
    "most", "more", "less", "likely", "relevant", "over", "other", "not", "no",
}

# Cues that the stated analysis has been *negated* — a faithful recommender should
# then prefer candidates that DON'T match the (now-inverted) signal.
_NEG_CUES = (
    "opposite", "dislike", "avoid", "hate", "not interested",
    "does not", "doesn't", "no longer", "instead of",
)


def _keywords(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) > 2 and t not in _STOP}


class MockModel:
    """Offline, deterministic, meaning-sensitive model for demos and tests.

    It never sees ground truth. It parses the candidate list from the prompt and
    ranks candidates by keyword overlap with the reasoning block (if present), else
    by overlap with the user's history. Editing the reasoning therefore changes the
    ranking in a meaning-tracking way — a stand-in for a *faithful* recommender.
    """

    def __call__(self, prompt: str, system: Optional[str] = None) -> str:
        cands = re.findall(r"^\[(\d+)\]\s*(.+)$", prompt, re.MULTILINE)  # [(num, label)]
        # A *supplied* analysis only ever appears under this heading (conditioned_prompt);
        # any bare <reason> elsewhere is just the format template in the elicitation prompt.
        analysis_m = re.search(
            r"analysis of the user's preferences:\s*<reason>(.*?)</reason>",
            prompt, re.DOTALL | re.IGNORECASE,
        )

        # Choose the signal to rank by: the supplied analysis, else the history block.
        if analysis_m is not None:
            analysis = analysis_m.group(1)
            signal = _keywords(analysis)
            # Honor stated polarity: a negated analysis inverts the ranking direction,
            # so flipping the reasoning (or any "opposite" cue) moves the pick.
            invert = any(cue in analysis.lower() for cue in _NEG_CUES)
        else:
            hist_m = re.search(r"history.*?:\s*(.*?)\n\n", prompt, re.DOTALL | re.IGNORECASE)
            signal = _keywords(hist_m.group(1) if hist_m else prompt)
            invert = False

        scored = []
        for num, label in cands:
            overlap = len(signal & _keywords(label))
            scored.append((int(num), overlap, label))
        # rank by overlap (desc normally, asc when the analysis is negated), then by
        # original index asc (stable, deterministic).
        sign = 1 if invert else -1
        scored.sort(key=lambda t: (sign * t[1], t[0]))
        ranking = "RANKING: " + ", ".join(str(num) for num, _, _ in scored)

        # If asked to reason first (elicitation prompt, no analysis supplied), emit a
        # <reason> echoing the history keywords so downstream edits have something to bite.
        if analysis_m is None and "<reason>" in prompt:
            hist_m = re.search(r"history.*?:\s*(.*?)\n\n", prompt, re.DOTALL | re.IGNORECASE)
            kws = sorted(_keywords(hist_m.group(1) if hist_m else ""))[:6]
            kw_str = ", ".join(kws) if kws else "general gaming"
            reason = f"<reason>The user likes {kw_str}; prefer those.</reason>"
            return reason + "\n" + ranking
        return ranking
