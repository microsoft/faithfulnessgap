# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quickstart — runs fully offline on MockModel, no API keys, no network.

    python examples/quickstart.py

MockModel is a deterministic stand-in that ranks candidates by keyword overlap with
the reasoning it is given, so editing the reasoning visibly moves its pick. It is
*meaning-sensitive by construction* — the point here is to show the audit end to end
and what a FAITHFUL signature looks like, not to make a claim about any real model.

Swap `MockModel()` for your own `model(prompt, system=None) -> str` callable (see
examples/openai_template.py) to audit a real model on your own cases.
"""
from faithgap import Case, FaithfulnessAudit, MockModel

# A handful of toy cases: each is a user's history, a candidate pool to rank, and the
# held-out "true" next item. Items are just strings here.
CASES = [
    Case(
        history=["Halo combat shooter", "Doom demon shooter", "Call of Duty war shooter"],
        candidates=["Stardew Valley farming sim", "Battlefield military shooter",
                    "Tetris puzzle blocks"],
        positive="Battlefield military shooter",
        user="u_shooter",
    ),
    Case(
        history=["Stardew Valley farming sim", "Animal Crossing village life",
                 "Harvest Moon farming"],
        candidates=["Doom demon shooter", "Story of Seasons farming sim",
                    "Gran Turismo racing"],
        positive="Story of Seasons farming sim",
        user="u_cozy",
    ),
    Case(
        history=["Gran Turismo racing", "Forza racing cars", "Need for Speed racing"],
        candidates=["Tetris puzzle blocks", "F1 racing simulator",
                    "Animal Crossing village life"],
        positive="F1 racing simulator",
        user="u_racing",
    ),
]


def main() -> None:
    audit = FaithfulnessAudit(model=MockModel())
    result = audit.run(CASES, bootstrap=2000)
    print(result.summary())
    print()
    print("As JSON:")
    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
