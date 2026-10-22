# RCL — Reasoning-Constraint Language

**A small DSL that forces LLM reasoning through a formal grammar, so every step is structured, traceable, and correctable.**

Built for the *Syntax Summit* hackathon. Runs on Gemma 4 through the `google-genai` SDK.

> **Honest scope:** RCL catches *structural incompleteness* and (with grounding validators) *checkable factual errors*. It does not detect semantic falsehood on its own, and it does not claim to "prevent hallucination."

---

## Why

Free-form chain-of-thought is hard to audit: you can't tell which step a conclusion rests on, how sure the model was, or what breaks if one fact turns out wrong. RCL makes the reasoning a tree with typed, validated nodes.

## The language

Seven keywords, indentation-based tree:

```
PROBLEM p1 [confidence: 0.7]: "..."
    CONSTRAINT c1 [confidence: 0.95] [cite: "case protocol"]: "..."
    REASON r1 [DECIDE_AT] [confidence: 0.8] [cite: "c1"]: "..."
        REASON r1a [confidence: 0.9] [cite: UNSUPPORTED]: "..."
    SOLUTION s1 [confidence: 0.85] [cite: "r1"]: "..."
    DELTA: "what would change this conclusion"
```

- Only `REASON` nests under `REASON`; `SOLUTION` / `DELTA` / `CONSTRAINT` are always top-level per `PROBLEM`.
- Every node (except `PROBLEM` / `DELTA`) can carry `[DECIDE_AT]`, `[confidence: float]`, `[cite: "..." | UNSUPPORTED]`.
- Subtree confidence propagates bottom-up with `min()` — a branch is only as strong as its weakest node.

Grammar: [`rcl_grammar.lark`](rcl_grammar.lark)

## Features

| Feature | What it does |
|---|---|
| Structural validator | Required fields, `DECIDE_AT` present, `SOLUTION` resolves (or names the dead-end branch), citations trace to real node ids |
| `RCLConfig` acceptance bar | Min/max reasons, tree depth, confidence thresholds, mandatory constraint coverage, max `UNSUPPORTED` ratio |
| Two-way authoring | Human writes `PROBLEM` + `CONSTRAINT`; the model fills `REASON` / `SOLUTION` / `DELTA` |
| `strict` / `open` modes | `strict` blocks the model from touching constraints (code-enforced); `open` lets it add constraints tagged `[model_added]` |
| Repair loop | Generate → splice → validate → retry (3x), feeding the exact validator error back over a persistent chat session |
| External validators | Pluggable `(node) -> ValidatorResult` hooks, `strict` (reject) or `advisory` (annotate) |
| Backprop | `update_node()` corrects one node, recomputes subtree confidence, flags dependent `SOLUTION`s as `needs_review` — no full regeneration |
| Linked problems | A later `PROBLEM` can cite an earlier `SOLUTION`; corrections ripple across problems |
| Visualization | Matplotlib tree: color = confidence, gold star = `DECIDE_AT`, dashed border = `UNSUPPORTED`, dashed blue = citations, rings = backprop ripple |

## Repository layout

```
rcl_grammar.lark          # the grammar
rcl_parser.py             # standalone parser + tree builder + validator
examples/
  production_demo.py      # fraud-investigation showcase + RCL vs CoT (clinical dosing)
  crime_scene_demo.py     # 500-suspect linked problems, real grounding validator, evidence-retraction backprop
docs/
  DESIGN_NOTES.md         # design decisions and limitations
requirements.txt
```

## Quick start

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt

# standalone parser sanity check
python rcl_parser.py

# demos (need a free key from https://aistudio.google.com/apikey)
export GEMINI_API_KEY="your-key"
python examples/crime_scene_demo.py
python examples/production_demo.py
```

Charts and tree images are written to `./outputs/` (or `/content` on Colab).
The demos also run in Google Colab: paste a script cell-by-cell and set `API_KEY`.

**Fast iteration:** in `examples/crime_scene_demo.py`, lower `NUM_SUSPECTS` (e.g. `25`) for quick test runs; keep `500` for the full demo.

## The demos

**Crime scene (grounded, linked).** 500 synthetic suspects live in a real dataset, not as tree nodes. Two linked `PROBLEM` blocks narrow the pool (alibi + motive, then opportunity + forensic); the second cites the first's solution. A grounding validator re-runs the actual filters against the actual data and, in `strict` mode, rejects wrong counts. A post-arrest scene retracts forensic evidence and `update_node_cross_problem()` flags exactly which conclusions depended on it.

**Production demo.** An all-features fraud-investigation showcase, plus an RCL-vs-CoT comparison on a hard dosing-math problem with real token and wall-clock accounting.

## Results, reported honestly

On the clinical dosing problem, RCL and CoT **tied at 6/6**. RCL is not pitched as "more accurate"; the tradeoff is **verifiability and correctability versus extra tokens and time**. Results vary per run; the demos print their own accounting.

## What RCL catches, and what it doesn't

**Catches:** missing fields, dangling citations, unresolved solutions, thin or over-confident reasoning, uncovered constraints, and any claim a grounding validator can check against real data.

**Doesn't catch:** semantically false claims with no validator behind them. The tree is a tree, not a DAG (v1 scope), and node ids are ASCII-only.

**Designed, not built:** branch/rollback, and a natural-language front-end.

## Real-world angle

Settings where *why* matters as much as *what*: health triage, post-conviction review, audit trails. When one input is later found faulty, RCL shows which conclusions rested on it.

## License

MIT — see [LICENSE](LICENSE).
