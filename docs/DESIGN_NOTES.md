# RCL Project — Status Summary (carry this into a new chat)

## What this is
RCL (Reasoning-Constraint Language) — a DSL forcing LLM reasoning through a formal
grammar (PROBLEM/CONSTRAINT/REASON/SOLUTION/DELTA), for a Devpost hackathon
("Syntax Summit"), deadline **Sep 5, 2026, 1pm GMT+1**. Solo entry.

## Core design (done, tested, working)
- Grammar: 7 keywords, indentation-based tree. Only REASON nests under REASON;
  SOLUTION/DELTA/CONSTRAINT always top-level per PROBLEM.
- Every node (except PROBLEM/DELTA): `[DECIDE_AT]? [confidence: float] [cite: str|UNSUPPORTED]`
- Subtree confidence = bottom-up min() propagation (a branch is only as strong as its weakest node) — confirmed working.
- Validator: structural checks (required fields, DECIDE_AT present, SOLUTION resolves or names dead-end branch, citations trace to real node ids) + configurable `RCLConfig` acceptance bar (min/max reasons, tree depth, confidence thresholds, mandatory constraint coverage, max UNSUPPORTED ratio).
- Two-way authoring: human writes PROBLEM+CONSTRAINT, model fills REASON/SOLUTION/DELTA. `strict` mode blocks model from touching constraints (code-enforced, not just prompted); `open` mode lets it add constraints tagged `[model_added]`.
- Generate→splice→validate→repair loop, 3 retries, feeds exact validator error back to model.
- External validator hook: `ValidatorResult`, pluggable `(node)->ValidatorResult` callables, `strict` (hard reject) or `advisory` (annotate) mode.
- Backprop: `update_node()` corrects one node, recomputes subtree confidence, flags dependent SOLUTIONs as `needs_review` — bounded blast radius, no full regen.
- Visualization: matplotlib tree render, color=confidence, gold star=DECIDE_AT, dashed border=UNSUPPORTED, dashed blue=citation links, ripple rings for backprop corrections.

## Known limitations (documented, intentional)
Catches structural incompleteness, not semantic falsehood. Tree not DAG (v1 scope decision). ASCII-only node IDs.

## Files that exist (in outputs)
- `rcl_grammar.lark`, `rcl_parser.py` — original standalone, 8/8 fixtures passed
- `rcl_colab_test.py` — first two-way authoring test, tested working
- `rcl_production_demo.py` — Gemma 4, token/time accounting, Part1 showcase (fraud investigation, all features) + Part2 RCL-vs-CoT on a hard clinical dosing math problem (both scored 6/6 — tied, not a CoT failure — this is honest data, reframed pitch around verifiability/cost tradeoff, not "CoT fails")
- `rcl_crime_scene_demo.py` — most advanced: 500-suspect dataset (real, seeded, NOT embedded as tree nodes), two LINKED PROBLEM blocks (p2 cites p1's solution — real relational reasoning, no grammar change needed), real grounding validator (re-runs actual filters against actual data, strict mode rejects wrong counts), post-arrest evidence-retraction backprop scene with cross-problem ripple checking (`update_node_cross_problem`)

## In-progress fix (interrupted, not yet completed)
User reported `rcl_crime_scene_demo.py` "takes too much time." Root cause identified:
every validator-triggered retry re-sends the ENTIRE prompt including embedded CSV data
(~3000 tokens) from scratch. Fix in progress: switch `generate_and_splice`'s retry loop
to a persistent `client.chats.create()` session (confirmed available in google-genai SDK)
so only the validator error gets sent on retry, not the full context again. Also adding:
live status/progress prints throughout, and making suspect count configurable (default
500, suggest lower for fast iteration).
**This fix was not finished — needs to be completed on rcl_crime_scene_demo.py.**

## Strategic decisions made
- Model: Gemma 4 (`gemma-4-31b-it`) via `google-genai` SDK, same API as Gemini.
- Chose NOT to build: branch/rollback, natural-language front-end (scoped as
  "designed not built" in README for time reasons).
- Chose NOT to build: HLE topology question demo (too hard to verify, risk of
  looking like dressed-up nonsense to an expert judge) — mentioned only as
  citation-of-difficulty-standard if it comes up, not a built feature.
- Pitch framing: don't oversell "prevents hallucination" — say "catches structural
  incompleteness, makes reasoning traceable and correctable," and be honest when
  CoT ties or wins on a given run rather than only showing favorable results.

## Remaining work before Sep 5
1. Finish the chat-session retry fix on rcl_crime_scene_demo.py
2. Run all three demo scripts for real, fix whatever breaks
3. Record demo video (5 min max)
4. Write submission README (thesis, comparison vs CoT/ReAct/guardrails, "what this
   catches vs doesn't" section, real-world tie-in — health triage / post-conviction
   review angle)
5. Submit with buffer day(s) before deadline
