# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Template: wrap a real OpenAI-compatible API as a faithgap model callable.

A faithgap *model* is any callable ``model(prompt, system=None) -> str``. That's the
entire contract — no base class. Below is a thin wrapper over an OpenAI-compatible
chat endpoint (works with OpenAI, Azure OpenAI, vLLM, Together, etc. by setting
OPENAI_BASE_URL). Adapt the body to whatever client you already use.

    pip install openai
    export OPENAI_API_KEY=sk-...
    # optional: export OPENAI_BASE_URL=... ; export OPENAI_MODEL=gpt-4o-mini
    python examples/openai_template.py

faithgap itself needs no keys; only this real-model example does.
"""
import os

from faithgap import Case, FaithfulnessAudit


class OpenAIModel:
    """Minimal ``model(prompt, system=None) -> str`` over an OpenAI-compatible API."""

    def __init__(self, model: str = None, temperature: float = 0.0):
        from openai import OpenAI  # imported lazily so the package stays dependency-free
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),  # None -> api.openai.com
        )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def __call__(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""


# Bring your own cases from your own dataset. These three are just to show the shape.
CASES = [
    Case(
        history=["The Legend of Zelda", "Hollow Knight", "Ori and the Blind Forest"],
        candidates=["Celeste", "FIFA 23", "Microsoft Flight Simulator"],
        positive="Celeste",
        user="u1",
    ),
    Case(
        history=["FIFA 22", "NBA 2K23", "Madden NFL"],
        candidates=["FIFA 23", "Disco Elysium", "Factorio"],
        positive="FIFA 23",
        user="u2",
    ),
]


def main() -> None:
    model = OpenAIModel()
    # trace="reason" audits the shown <reason>; use trace="think" for a reasoning
    # model's hidden <think> CoT. edits="llm" asks the model to do the flip/paraphrase
    # rewrites (higher quality, costs extra calls) instead of the offline heuristics.
    audit = FaithfulnessAudit(model=model, trace="reason", edits="heuristic")
    result = audit.run(CASES)
    print(result.summary())


if __name__ == "__main__":
    main()
