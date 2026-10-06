# faithgap

### [Accepted in RecSys 2026] — *How Faithful Is the Reasoning of LLM Recommenders? A Counterfactual Audit*

**Is your LLM recommender's reasoning actually driving its picks — or just decorating them?**

`faithgap` is a small, bring-your-own-model tool that measures the **faithfulness gap**
of an LLM recommender: how much more its recommendation moves when you change the
*meaning* of its stated reasoning than when you merely *reword* it.

It is the method from the paper **"How Faithful Is the Reasoning of LLM Recommenders?
A Counterfactual Audit"** (RecSys 2026), packaged as a reusable library you can point at
any model and any dataset.

---

## The idea in one paragraph

Ask a model to reason-then-recommend. Take its stated reasoning *R*, get its ranking
conditioned on *R* (the **base** pick). Now edit *R* and re-rank:

| edit | what it does | type |
|------|--------------|------|
| **flip** | reverses the stated preference | meaning change |
| **swap** | substitutes another user's reasoning | meaning change |
| **erase** | removes the reasoning entirely | meaning change |
| **paraphrase** | rewords it, same meaning | **control** |

> **faithfulness gap = mean(top-1 change rate under flip / swap / erase) − change rate under paraphrase**

If the pick is really produced *by* the reasoning, meaning-changing edits move it while
the paraphrase control leaves it put → **large positive gap (faithful)**. If the pick
barely budges when the meaning is destroyed → **gap near zero (decorative)**.

### The metric

Let the meaning-changing edits be *M* = {flip, swap, erase} and the control be
*paraphrase*. For a set of cases, let **Δ(e)** be the fraction of cases whose audited
decision changes when edit *e* is applied to the reasoning. Then

$$\mathrm{FG} \;=\; \underbrace{\frac{1}{|M|}\sum_{e \in M} \Delta(e)}_{\text{mean change under meaning edits}} \;-\; \underbrace{\Delta(\text{paraphrase})}_{\text{control}}$$

Equivalently, per case *i* and edit *e*, with δ\_i(e) = 𝟙[decision changed]:

```
FG = mean over (case, meaning-edit) pairs of δ  −  mean over cases of δ(paraphrase)
```

FG ∈ [−1, 1]. The paraphrase term subtracts off decision churn that has nothing to do
with meaning (model nondeterminism, prompt sensitivity), so FG isolates the part of the
behavior that is *causally tied to what the reasoning says*. The **decision** whose
change we count defaults to the top-1 item but is configurable (see below).

---

## Install

```bash
git clone https://github.com/microsoft/faithfulnessgap.git && cd faithfulnessgap
pip install -e .
```

Zero runtime dependencies — pure standard library. (Python ≥ 3.9.)

## Quickstart (offline, no API keys)

```python
from faithgap import FaithfulnessAudit, Case, MockModel

cases = [
    Case(history=["Halo shooter", "Doom shooter"],
         candidates=["Battlefield shooter", "Stardew farming", "Tetris puzzle"],
         positive="Battlefield shooter"),
]

result = FaithfulnessAudit(model=MockModel()).run(cases)
print(result.summary())
print(result.faithfulness_gap)   # a single number in [-1, 1]
```

Run the bundled demo:

```bash
python examples/quickstart.py
pytest -q          # smoke tests, all offline
```

`MockModel` is a deterministic, meaning-sensitive stand-in so the quickstart runs with
no network and no keys. It is there to show the audit end to end — not to make a claim
about any real system.

## Audit your own model

A *model* is any callable `model(prompt, system=None) -> str`. Wrap whatever you have:

```python
def my_model(prompt, system=None):
    return call_your_llm(system_prompt=system, user_prompt=prompt)

audit = FaithfulnessAudit(
    model=my_model,
    trace="reason",     # "reason" = the shown <reason>; "think" = hidden <think> CoT
    edits="heuristic",  # "heuristic" (offline) or "llm" (model rewrites flip/paraphrase)
)
result = audit.run(my_cases, bootstrap=1000)
print(result.summary())
```

See [`examples/openai_template.py`](examples/openai_template.py) for a complete
OpenAI-compatible wrapper (also works with Azure OpenAI / vLLM / Together via
`OPENAI_BASE_URL`).

Build a `Case` from your own dataset — `history` (past items, oldest→newest),
`candidates` (the pool to rank, including the held-out next item), and `positive`
(that held-out item). Items can be strings, or ids plus a `titles={id: title}` map on
the audit.

## Reading the result

```
Faithfulness audit  (n=3 cases)
  faithfulness gap     : +0.667   95% CI [+0.000, +1.000]
  meaning-change rate  : 0.667   (flip / swap / erase)
  paraphrase control   : 0.000   (reword, same meaning)
  per-op #1-change rate:
      flip       : 0.667
      swap       : 0.667
      erase      : 0.667
      paraphrase : 0.000 (control)
  verdict              : FAITHFUL
```

`result.to_dict()` gives you the same numbers as JSON. The **verdict** is a coarse label
(`faithful` ≥ 0.25, `decorative` ≤ 0.10, else `mixed`); the gap and its bootstrap CI are
the quantities to report.

## Choosing the decision to audit (beyond top-1)

By default the gap counts whether the **top-1 pick** changed. That readout is pluggable —
pass `decision=f`, a function `f(ranking, case) -> value` whose change is what gets
measured:

```python
# top-3 set (order-sensitive): did any of the first three move?
FaithfulnessAudit(model=m, decision=lambda rank, case: tuple(rank[:3]))

# rank of the held-out ground-truth item: did the reasoning move *it*?
FaithfulnessAudit(model=m, decision=lambda rank, case: rank.index(case.positive))
```

A one-argument `decision(ranking)` is also accepted. This is the knob that unties the
audit from "top pick only."

## What it generalizes to

The **method** is general; this package is one concrete instantiation of it.

- **Domain-agnostic core.** The pattern — *edit the explanation, keep a paraphrase as a
  same-meaning control, measure how much the decision moves* — is a behavioral/causal
  faithfulness test for **any** natural-language rationale. The edit set, the gap formula,
  and the bootstrap don't know anything about recommenders.
- **Reusable as-is** for any model that reason-then-recommends over a candidate list:
  different domains (movies, products, music), `trace="reason"` vs `trace="think"`
  (a reasoning model's hidden CoT), and now any `decision` readout (top-k, rank-of-item).
- **What you'd swap for a different task.** The two prompts in `prompts.py`
  (`reason_prompt` = elicit the explanation; `conditioned_prompt` = decide *given* an
  explanation) are ranking-specific. To audit, say, an explained **classifier** or a
  QA/explanation system, replace those two prompts and the `decision` readout; the edits,
  the gap math, and the CI carry over unchanged.
- **What falls outside this instantiation.** The edits here operate on a **written**
  explanation. Explanations that aren't prose — a feature-attribution map from SHAP / LIME,
  or attention weights — have no sentence to flip, swap, or reword, so the text edits in
  `edits.py` don't apply directly. The *approach* still carries over: perturb the
  explanation, measure how much the decision moves, and keep a do-nothing perturbation as
  the control. You'd just swap the text editor for one that edits that artifact (e.g.
  ablate the top-weighted features, with low-weight features as the null control).

## What this is and isn't

- It measures **behavioral** faithfulness — whether the decision is causally sensitive to
  the stated reasoning — not whether the reasoning is "correct" or plausible.
- A positive gap says the stated trace *matters*; a near-zero gap says it's decorative.
  Neither is a verdict on model quality.
- The heuristic edits are deliberately simple and transparent. `edits="llm"` uses the
  audited model to produce higher-quality flip/paraphrase rewrites at the cost of extra
  calls.

## Cite

The methodology and the full study (models, datasets, and headline numbers) are in the
paper; this repository is that method as a reusable tool.

```bibtex
@inproceedings{shah2026faithful,
  title     = {How Faithful Is the Reasoning of LLM Recommenders? A Counterfactual Audit},
  author    = {Shah, Arpita Vasant},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)},
  year      = {2026},
  doi       = {10.1145/3773078.3841294}
}
```

## License

MIT — see [LICENSE.txt](LICENSE.txt). Provided as-is, for research use. It does not bundle
any dataset; bring your own cases.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause
confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are
subject to those third-party's policies.
