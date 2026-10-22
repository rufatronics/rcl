# RCL — Production Demo: Gemma 4, Token/Time Accounting, RCL vs. CoT Head-to-Head
# ============================================================
# HOW TO USE IN COLAB:
#   1. New Colab notebook, upload this file OR paste cell-by-cell
#   2. Run the install cell first
#   3. Paste your Gemini/AI Studio API key into API_KEY below
#      (get one free at https://aistudio.google.com/apikey)
#   4. Run the rest top to bottom
#
# MODEL: Gemma 4 (gemma-4-31b-it) — served through the same Gemini API /
# google-genai SDK, but Gemma is open-weight, so it runs at $0/token on the
# API rather than the metered Gemini free-tier quota. Swap MODEL below to
# "gemma-4-26b-a4b-it" (MoE, lighter/faster) if you hit capacity issues.
#
# WHAT THIS RUNS:
#   PART 1 — Showcase problem: one long, multi-constraint incident-response
#            problem exercising every structural feature at once (strict +
#            open mode, DECIDE_AT, UNSUPPORTED, model_added constraints,
#            citation validator, backprop correction + ripple). All trees
#            are visualized AND saved as downloadable PNGs.
#   PART 2 — RCL vs. Chain-of-Thought head-to-head on a verifiable math
#            problem: same problem, two approaches, real token counts and
#            wall-clock time recorded for both, correctness checked against
#            a hand-computed ground truth, results rendered as a comparison
#            chart and downloaded.
# ============================================================

# %% [install]
# !pip install google-genai lark -q

# %% [imports]
import re
import time
from pathlib import Path
from lark import Lark
from lark.indenter import Indenter
from google import genai

# %% [config]
import os
API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_KEY_HERE")   # https://aistudio.google.com/apikey
MODEL = "gemma-4-31b-it"           # dense Gemma 4 — swap to "gemma-4-26b-a4b-it" for MoE/lighter
MAX_RETRIES = 3
NUM_SUSPECTS = 500                 # lower this (e.g. 25) for fast iteration while testing;
                                    # 500 is the "real" run for the recorded demo/video

client = genai.Client(api_key=API_KEY)

# Output folder for charts/trees: /content on Colab, ./outputs elsewhere
OUT_DIR = "/content" if os.path.isdir("/content") else "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_usage(response) -> dict:
    """Pulls token counts out of the API response. Field names have moved
    around across SDK versions, so this tries the current names first and
    falls back gracefully to zeros rather than crashing a demo run over a
    metadata field rename."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    prompt = getattr(meta, "prompt_token_count", None) or 0
    output = getattr(meta, "candidates_token_count", None) or 0
    total = getattr(meta, "total_token_count", None) or (prompt + output)
    return {"prompt_tokens": prompt, "output_tokens": output, "total_tokens": total}


def download_if_colab(path: str):
    """Triggers a browser download in Colab; no-ops with a clear message
    outside Colab (e.g. running this locally) rather than failing."""
    try:
        from google.colab import files
        files.download(path)
    except ImportError:
        print(f"(Not running in Colab — file saved locally at {path})")
    except Exception as e:
        print(f"(Download trigger failed, file is still saved at {path}: {e})")

# %% [grammar]
# Embedded inline (not read from disk) so this file is self-contained for Colab.
RCL_GRAMMAR = r'''
start: problem+

problem: "PROBLEM" NODE_ID prob_tags ":" STRING _NL _INDENT block+ _DEDENT

block: constraint
     | reason
     | solution
     | delta

constraint: "CONSTRAINT" NODE_ID tags ":" STRING _NL

reason: "REASON" NODE_ID tags ":" STRING _NL [_INDENT reason+ _DEDENT]

solution: "SOLUTION" NODE_ID tags ":" STRING _NL

delta: "DELTA" ":" STRING _NL

prob_tags: prob_tag*
prob_tag: confidence

tags: tag*
tag: decide_at
   | confidence
   | cite
   | model_added

decide_at:   "[" "DECIDE_AT" "]"
confidence:  "[" "confidence" ":" FLOAT "]"
cite:        "[" "cite" ":" (STRING | UNSUPPORTED) "]"
model_added: "[" "model_added" "]"

UNSUPPORTED: "UNSUPPORTED"

NODE_ID: /[a-zA-Z_][a-zA-Z0-9_]*/
FLOAT: /[0-9]+(\.[0-9]+)?/
STRING: /"[^"]*"/

%declare _INDENT _DEDENT
_NL: /(\r?\n[\t ]*)+/

%ignore /[ \t]+/
'''
# NOTE: this adds one new tag vs. the original grammar file — `model_added`,
# a bare marker tag (like DECIDE_AT) used only in `open` mode so any
# CONSTRAINT the model adds is visibly distinguishable from human-authored
# ones. It's additive; strict-mode files never use it.

# %% [indenter]
class RCLIndenter(Indenter):
    NL_type = "_NL"
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4

_parser = Lark(RCL_GRAMMAR, parser="lalr", postlex=RCLIndenter())

# %% [tree model]
class Node:
    def __init__(self, kind, node_id, text, decide_at=False,
                 confidence=None, cite=None, model_added=False, parent=None):
        self.kind = kind
        self.node_id = node_id
        self.text = text
        self.decide_at = decide_at
        self.confidence = confidence
        self.cite = cite
        self.model_added = model_added
        self.parent = parent
        self.children = []
        self.subtree_confidence = None

    def __repr__(self):
        tag = " [model_added]" if self.model_added else ""
        return f"<{self.kind} {self.node_id} conf={self.confidence}{tag}>"


class ParsedProblem:
    def __init__(self, problem_node, constraints, reason_roots, solutions, delta_text):
        self.problem = problem_node
        self.constraints = constraints
        self.reason_roots = reason_roots
        self.solutions = solutions
        self.delta_text = delta_text


class RCLError(Exception):
    pass


# %% [config]
from dataclasses import dataclass, field


@dataclass
class RCLConfig:
    """Tunable acceptance criteria — separate from grammar validity.
    Grammar validity asks 'is this well-formed'. This asks 'does this meet
    the bar we actually want for this deployment' — a compliance tool and a
    brainstorming tool should not share the same thresholds."""

    mode: str = "strict"                       # "strict" | "open"

    min_reasons: int = 1                        # fewest REASON nodes allowed per PROBLEM
    max_reasons: int | None = None              # None = unbounded

    min_tree_depth: int = 1                     # shallowest allowed REASON nesting
    max_tree_depth: int | None = None            # None = unbounded

    min_confidence_threshold: float = 0.0        # every SOLUTION must meet/exceed this
    min_subtree_confidence: float | None = None  # if set, every top-level REASON's
                                                   # subtree_confidence must meet/exceed this
                                                   # (catches "strong own claim, weak branch")

    require_constraint_coverage: bool = True      # every human CONSTRAINT must be referenced
                                                   # by id in at least one REASON/SOLUTION cite

    max_unsupported_ratio: float | None = None    # e.g. 0.5 = at most half of REASON nodes
                                                   # may be [cite: UNSUPPORTED]; None = no cap

    validators: list = field(default_factory=list)  # list of (name, callable) — see ValidatorResult
                                                       # applied to any node's [cite: ...] field
    solution_validators: list = field(default_factory=list)  # list of (name, callable) — applied
                                                       # to a SOLUTION node's .text directly, for
                                                       # domain checks like "is this answer actually
                                                       # correct" rather than "is this citation real"
    validator_mode: str = "strict"                    # "strict" = failing validator raises RCLError
                                                        # "advisory" = failing validator just annotates
                                                        # the node with a note, doesn't block validation

    def describe(self) -> str:
        """Human-readable summary, used to tell the model what bar it's being held to."""
        lines = [f"- mode: {self.mode}"]
        lines.append(f"- must contain between {self.min_reasons} and "
                      f"{self.max_reasons or 'unlimited'} REASON node(s) total")
        lines.append(f"- reasoning tree depth must be between {self.min_tree_depth} and "
                      f"{self.max_tree_depth or 'unlimited'} level(s)")
        if self.min_confidence_threshold > 0:
            lines.append(f"- every SOLUTION must have confidence >= {self.min_confidence_threshold}")
        if self.min_subtree_confidence is not None:
            lines.append(f"- every reasoning branch's weakest link (subtree confidence) "
                          f"must be >= {self.min_subtree_confidence}")
        if self.require_constraint_coverage:
            lines.append("- every CONSTRAINT given must be explicitly referenced by id "
                          "in at least one REASON or SOLUTION's [cite: ...]")
        if self.max_unsupported_ratio is not None:
            lines.append(f"- at most {self.max_unsupported_ratio:.0%} of REASON nodes may "
                          f"be [cite: UNSUPPORTED] — most claims need a real source")
        if self.validators:
            names = ", ".join(name for name, _ in self.validators)
            lines.append(f"- citations will be checked by external validator(s): {names}. "
                          f"Do not invent a citation that would not pass a real source check.")
        if self.solution_validators:
            names = ", ".join(name for name, _ in self.solution_validators)
            lines.append(f"- your final SOLUTION will be checked for correctness by: {names}. "
                          f"Compute carefully — a plausible-sounding wrong answer will be caught.")
        return "\n".join(lines)


def _unquote(tok):
    return str(tok)[1:-1]


def _extract_tags(tags_tree):
    decide_at = False
    confidence = None
    cite = None
    model_added = False
    seen = set()
    for tag in tags_tree.children:
        inner = tag.children[0]
        if inner.data in seen:
            raise RCLError(f"duplicate [{inner.data}] tag on the same node.")
        seen.add(inner.data)
        if inner.data == "decide_at":
            decide_at = True
        elif inner.data == "model_added":
            model_added = True
        elif inner.data == "confidence":
            confidence = float(inner.children[0])
        elif inner.data == "cite":
            val = inner.children[0]
            cite = "UNSUPPORTED" if str(val) == "UNSUPPORTED" else _unquote(val)
    return decide_at, confidence, cite, model_added


def _build_reason(tree, parent):
    node_id = str(tree.children[0])
    decide_at, confidence, cite, model_added = _extract_tags(tree.children[1])
    text = _unquote(tree.children[2])
    node = Node("REASON", node_id, text, decide_at, confidence, cite, model_added, parent)
    for child in tree.children[3:]:
        if child.data == "reason":
            node.children.append(_build_reason(child, node))
    return node


def build_tree(parse_tree):
    problems = []
    for prob_tree in parse_tree.children:
        node_id = str(prob_tree.children[0])
        _, confidence, _, _ = _extract_tags(prob_tree.children[1])
        text = _unquote(prob_tree.children[2])
        problem_node = Node("PROBLEM", node_id, text, False, confidence, None, False, None)

        constraints, reason_roots, solutions, delta_text = [], [], [], None

        for block in prob_tree.children[3:]:
            inner = block.children[0]
            if inner.data == "delta":
                delta_text = _unquote(inner.children[0])
                continue
            if inner.data == "constraint":
                cid = str(inner.children[0])
                da, conf, cite, ma = _extract_tags(inner.children[1])
                ctext = _unquote(inner.children[2])
                constraints.append(Node("CONSTRAINT", cid, ctext, da, conf, cite, ma, problem_node))
            elif inner.data == "reason":
                reason_roots.append(_build_reason(inner, problem_node))
            elif inner.data == "solution":
                sid = str(inner.children[0])
                da, conf, cite, ma = _extract_tags(inner.children[1])
                stext = _unquote(inner.children[2])
                solutions.append(Node("SOLUTION", sid, stext, da, conf, cite, ma, problem_node))

        problems.append(ParsedProblem(problem_node, constraints, reason_roots, solutions, delta_text))
    return problems


# %% [validator]
def _all_reason_nodes(roots):
    out, stack = [], list(roots)
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def compute_subtree_confidence(node):
    if not node.children:
        node.subtree_confidence = node.confidence if node.confidence is not None else 1.0
        return node.subtree_confidence
    child_min = min(compute_subtree_confidence(c) for c in node.children)
    own = node.confidence if node.confidence is not None else 1.0
    node.subtree_confidence = min(own, child_min)
    return node.subtree_confidence


def check_decide_at(pp, all_reasons):
    if not any(n.decide_at for n in all_reasons + pp.constraints + pp.solutions):
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: no [DECIDE_AT] marker found anywhere in the tree."
        )


def check_required_fields(pp, all_reasons):
    for n in pp.constraints + all_reasons + pp.solutions:
        if n.confidence is None:
            raise RCLError(f"{n.kind} {n.node_id}: missing required [confidence: ...] field.")
        if n.cite is None:
            raise RCLError(f"{n.kind} {n.node_id}: missing required [cite: ...] field.")


def check_terminal_solution(pp):
    if not pp.solutions:
        leaves = [n for n in _all_reason_nodes(pp.reason_roots) if not n.children]
        dead_ends = ", ".join(f'{n.node_id} ("{n.text[:40]}...")' for n in leaves) or "no REASON nodes present"
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: unresolved — no SOLUTION found. "
            f"Dead-end leaf branch(es): {dead_ends}"
        )
    all_ids = {n.node_id for n in pp.constraints} | {n.node_id for n in _all_reason_nodes(pp.reason_roots)}
    for sol in pp.solutions:
        if sol.cite not in ("UNSUPPORTED", None) and sol.cite not in all_ids:
            found = any(re.search(rf"\b{re.escape(nid)}\b", sol.cite) for nid in all_ids)
            if not found:
                raise RCLError(
                    f"SOLUTION {sol.node_id}: cite \"{sol.cite}\" does not trace back to any "
                    f"node in PROBLEM {pp.problem.node_id}'s tree."
                )


def check_reason_count(pp, config: RCLConfig):
    count = len(_all_reason_nodes(pp.reason_roots))
    if count < config.min_reasons:
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: only {count} REASON node(s) present, "
            f"minimum required is {config.min_reasons}. Add more reasoning steps."
        )
    if config.max_reasons is not None and count > config.max_reasons:
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: {count} REASON node(s) present, "
            f"maximum allowed is {config.max_reasons}. Simplify the reasoning tree."
        )


def _tree_depth(node) -> int:
    if not node.children:
        return 1
    return 1 + max(_tree_depth(c) for c in node.children)


def check_tree_depth(pp, config: RCLConfig):
    if not pp.reason_roots:
        return
    depth = max(_tree_depth(r) for r in pp.reason_roots)
    if depth < config.min_tree_depth:
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: reasoning depth is {depth}, "
            f"minimum required is {config.min_tree_depth}. Branch deeper before concluding."
        )
    if config.max_tree_depth is not None and depth > config.max_tree_depth:
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: reasoning depth is {depth}, "
            f"maximum allowed is {config.max_tree_depth}. Flatten the reasoning tree."
        )


def check_confidence_thresholds(pp, config: RCLConfig):
    for sol in pp.solutions:
        if sol.confidence is not None and sol.confidence < config.min_confidence_threshold:
            raise RCLError(
                f"SOLUTION {sol.node_id}: confidence {sol.confidence} is below the "
                f"required minimum of {config.min_confidence_threshold}. Either strengthen "
                f"the supporting reasoning or do not conclude with SOLUTION yet."
            )
    if config.min_subtree_confidence is not None:
        for root in pp.reason_roots:
            if root.subtree_confidence is not None and root.subtree_confidence < config.min_subtree_confidence:
                raise RCLError(
                    f"REASON {root.node_id}: subtree_confidence {root.subtree_confidence} "
                    f"is below the required minimum of {config.min_subtree_confidence}. "
                    f"This branch has a weak link dragging down the whole chain — "
                    f"strengthen it or remove that branch."
                )


def check_constraint_coverage(pp, config: RCLConfig):
    if not config.require_constraint_coverage:
        return
    citers = _all_reason_nodes(pp.reason_roots) + pp.solutions
    for c in pp.constraints:
        referenced = any(
            n.cite and re.search(rf"\b{re.escape(c.node_id)}\b", n.cite)
            for n in citers
        )
        if not referenced:
            raise RCLError(
                f"CONSTRAINT {c.node_id} is never referenced by id in any REASON or "
                f"SOLUTION's [cite: ...]. Every given constraint must be explicitly "
                f"addressed, not just implicitly assumed."
            )


def check_unsupported_ratio(pp, config: RCLConfig):
    if config.max_unsupported_ratio is None:
        return
    reasons = _all_reason_nodes(pp.reason_roots)
    if not reasons:
        return
    unsupported = sum(1 for r in reasons if r.cite == "UNSUPPORTED")
    ratio = unsupported / len(reasons)
    if ratio > config.max_unsupported_ratio:
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: {ratio:.0%} of REASON nodes are "
            f"[cite: UNSUPPORTED], exceeding the allowed maximum of "
            f"{config.max_unsupported_ratio:.0%}. Provide real sources for more claims."
        )


# %% [external validator hook]
@dataclass
class ValidatorResult:
    """Return type an external validator callback must produce. A validator
    is any callable(node) -> ValidatorResult — this is the extension point
    where a real RAG grounding check, a fact-checking API, or a human review
    queue plugs in. RCL itself only checks structure; a validator is how you
    plug in something that checks truth."""
    passed: bool
    message: str = ""


def check_external_validators(pp, config: RCLConfig):
    if not config.validators:
        return
    citable_nodes = pp.constraints + _all_reason_nodes(pp.reason_roots) + pp.solutions
    for node in citable_nodes:
        if not node.cite or node.cite == "UNSUPPORTED":
            continue  # nothing to validate — UNSUPPORTED is already an honest admission
        for name, fn in config.validators:
            result = fn(node)
            if not result.passed:
                if config.validator_mode == "strict":
                    raise RCLError(
                        f"{node.kind} {node.node_id}: external validator '{name}' rejected "
                        f"citation \"{node.cite}\": {result.message}"
                    )
                else:  # advisory — annotate, don't block
                    node.validator_notes = getattr(node, "validator_notes", [])
                    node.validator_notes.append(f"[{name}] {result.message}")


def check_solution_validators(pp, config: RCLConfig):
    """Runs domain-specific correctness checks against a SOLUTION's stated
    text — separate from check_external_validators, which only inspects
    citations. This is where a real ground-truth check (like the math
    validator) plugs in: it answers 'is the conclusion actually right',
    which citation-checking alone can never answer."""
    if not config.solution_validators:
        return
    for sol in pp.solutions:
        for name, fn in config.solution_validators:
            result = fn(sol)
            if not result.passed:
                if config.validator_mode == "strict":
                    raise RCLError(
                        f"SOLUTION {sol.node_id}: solution validator '{name}' rejected "
                        f"the conclusion: {result.message}"
                    )
                else:
                    sol.validator_notes = getattr(sol, "validator_notes", [])
                    sol.validator_notes.append(f"[{name}] {result.message}")


def validate(pp, config: RCLConfig = None):
    config = config or RCLConfig()
    all_reasons = _all_reason_nodes(pp.reason_roots)
    check_required_fields(pp, all_reasons)
    check_decide_at(pp, all_reasons)
    check_terminal_solution(pp)
    for root in pp.reason_roots:
        compute_subtree_confidence(root)
    # structural checks above must pass before these — they assume a
    # well-formed, resolved tree with subtree_confidence already computed
    check_reason_count(pp, config)
    check_tree_depth(pp, config)
    check_confidence_thresholds(pp, config)
    check_constraint_coverage(pp, config)
    check_unsupported_ratio(pp, config)
    check_external_validators(pp, config)
    check_solution_validators(pp, config)
    return []


# %% [backpropagation: correcting a node ripples the consequence upward]
def find_node(pp, node_id):
    """Looks up any node in the tree by id — CONSTRAINT, REASON, SOLUTION, or PROBLEM."""
    if pp.problem.node_id == node_id:
        return pp.problem
    for n in pp.constraints + _all_reason_nodes(pp.reason_roots) + pp.solutions:
        if n.node_id == node_id:
            return n
    return None


def update_node(pp, node_id: str, new_confidence: float = None, new_cite: str = None,
                 update_reason: str = "") -> dict:
    """Corrects one node's declared confidence and/or citation, then propagates
    the consequence upward through the tree — like backpropagation, but for
    trust rather than gradients. This does NOT regenerate or delete anything;
    it recomputes subtree_confidence for every REASON root and flags any
    SOLUTION whose citation chain passes through the updated node as
    needing review, so the blast radius of one correction is always visible
    and bounded, never a silent full-tree reset.

    Returns a report: which subtree confidences changed, and which
    SOLUTIONs got flagged. Re-run validate(pp, config) afterward if you want
    to know whether the correction now violates an acceptance threshold."""
    node = find_node(pp, node_id)
    if node is None:
        raise RCLError(f"update_node: no node with id '{node_id}' found in PROBLEM {pp.problem.node_id}")

    old_confidence, old_cite = node.confidence, node.cite
    if new_confidence is not None:
        node.confidence = new_confidence
    if new_cite is not None:
        node.cite = new_cite

    node.update_log = getattr(node, "update_log", [])
    node.update_log.append({
        "reason": update_reason,
        "old_confidence": old_confidence, "new_confidence": node.confidence,
        "old_cite": old_cite, "new_cite": node.cite,
    })

    subtree_changes = {}
    for root in pp.reason_roots:
        before = root.subtree_confidence
        after = compute_subtree_confidence(root)
        if before != after:
            subtree_changes[root.node_id] = (before, after)

    # affected ids = the updated node itself, plus everything beneath it if it's a REASON
    affected_ids = {node_id}
    if node.kind == "REASON":
        affected_ids |= {n.node_id for n in _all_reason_nodes([node])}

    flagged_solutions = []
    for sol in pp.solutions:
        if sol.cite and any(re.search(rf"\b{re.escape(aid)}\b", sol.cite) for aid in affected_ids):
            sol.needs_review = True
            flagged_solutions.append(sol.node_id)

    return {
        "updated_node": node_id,
        "subtree_confidence_changes": subtree_changes,
        "flagged_solutions": flagged_solutions,
    }


MOCK_KNOWLEDGE_BASE = {
    "SLA doc", "Datadog metrics", "SQL slow query logs", "Network telemetry",
    "Database logs", "EXPLAIN plan", "CloudWatch logs", "WAF logs",
}


def mock_rag_validator(node) -> "ValidatorResult":
    if node.cite in MOCK_KNOWLEDGE_BASE:
        return ValidatorResult(True, "found in knowledge base")
    return ValidatorResult(
        False,
        f"'{node.cite}' does not match any known source in the knowledge base — "
        f"possible fabricated or unverifiable citation"
    )


def try_parse_and_validate(source: str, config: RCLConfig = None):
    """Returns (ok: bool, problems_or_error)."""
    try:
        tree = _parser.parse(source)
        problems = build_tree(tree)
        for pp in problems:
            validate(pp, config)
        return True, problems
    except RCLError as e:
        return False, f"RCLError: {e}"
    except Exception as e:
        return False, f"ParseError: {type(e).__name__}: {e}"


# %% [prompt + generation]
STRICT_INSTRUCTIONS = (
    "You may ONLY add REASON, SOLUTION, and DELTA blocks under the existing "
    "PROBLEM and CONSTRAINT(s) shown below. Do NOT add, remove, or modify any "
    "CONSTRAINT. Do NOT output the PROBLEM or CONSTRAINT lines again — only "
    "output the new lines to append, indented one level deeper than CONSTRAINT "
    "(4 spaces per level, matching the existing indentation)."
)

OPEN_INSTRUCTIONS = (
    "You may add REASON, SOLUTION, and DELTA blocks under the existing PROBLEM "
    "and CONSTRAINT(s) shown below. If you judge the given CONSTRAINT(s) to be "
    "incomplete for solving the PROBLEM, you may also add new CONSTRAINT nodes — "
    "but every CONSTRAINT you add must carry a [model_added] tag so it is never "
    "confused with a human-authored constraint. Do NOT modify or remove the "
    "human's existing CONSTRAINT(s). Do NOT output the PROBLEM line again — "
    "only output the new lines to append, indented one level deeper than "
    "PROBLEM (4 spaces per level, matching the existing indentation)."
)

RCL_SPEC = '''
RCL syntax reference (every node except PROBLEM/DELTA follows this shape):

  KEYWORD id [DECIDE_AT]? [confidence: FLOAT] [cite: "text"|UNSUPPORTED]: "content"

Rules:
- Every CONSTRAINT, REASON, and SOLUTION MUST have both [confidence: ...] and [cite: ...].
- Use [cite: UNSUPPORTED] (no quotes) if you have no real source — never invent a fake citation.
- At least one node in the whole tree must carry [DECIDE_AT] — mark the single
  most important branching/decision point with it.
- REASON nodes can nest (indent further) to branch into sub-reasoning.
- IMPORTANT: SOLUTION, DELTA, and CONSTRAINT are NEVER indented under a
  REASON. Even after several levels of nested REASON, SOLUTION and DELTA
  always return to the SAME indentation as the first CONSTRAINT you were
  given — they close out the whole PROBLEM, not a specific REASON branch.
- SOLUTION's [cite: ...] must reference an existing node id in the tree (e.g. "r3, some detail"),
  not something from outside the tree.
- Every PROBLEM must end with a DELTA line describing what changed, at the
  same indent level as CONSTRAINT/REASON/SOLUTION.
- Indentation is 4 spaces per level, exactly like Python.

Example of a complete, valid PROBLEM:

PROBLEM p1 [confidence: 0.7]: "Patient presents with fatigue and joint pain"
    CONSTRAINT c1 [confidence: 0.9] [cite: "clinical guideline"]: "Must rule out autoimmune cause"
    REASON r1 [DECIDE_AT] [confidence: 0.8] [cite: "Lupus Foundation guidelines"]: "Symptom pattern matches inflammatory marker profile"
        REASON r2 [confidence: 0.4] [cite: UNSUPPORTED]: "Could also indicate chronic fatigue syndrome"
        REASON r3 [confidence: 0.85] [cite: "ANA test result"]: "Positive ANA supports autoimmune pathway over CFS"
    SOLUTION s1 [confidence: 0.85] [cite: "r3, ANA test result"]: "Recommend rheumatology referral, r3 branch supported"
    DELTA: "Initial hypothesis (CFS) downgraded after r3; added referral action"
'''


def build_initial_prompt(human_block: str, mode: str, config: RCLConfig) -> str:
    """The one and only 'heavy' message for a generate_and_splice call — spec,
    instructions, config, and the human block (which may embed a large CSV).
    Sent exactly once per call, as turn 1 of a chat session."""
    instructions = STRICT_INSTRUCTIONS if mode == "strict" else OPEN_INSTRUCTIONS
    prompt = (
        f"{RCL_SPEC}\n\n{instructions}\n\n"
        f"Additional acceptance criteria for this task (your output will be "
        f"rejected if it doesn't meet these, in addition to the base grammar rules):\n"
        f"{config.describe()}\n\n"
        f"Human-authored input so far:\n\n{human_block}\n"
    )
    prompt += "\nOutput ONLY the RCL lines to append. No explanation, no markdown fences."
    return prompt


def build_retry_prompt(prior_error: str) -> str:
    """The 'light' follow-up message for retries. Because this is sent on an
    existing chat session (see generate_and_splice), the model still has the
    full spec/instructions/CSV from turn 1 in its context — we don't pay to
    re-transmit ~3000 tokens of case data on every retry, only this short
    correction message."""
    return (
        f"Your previous attempt failed validation with this exact error:\n"
        f"{prior_error}\n"
        f"Fix this specific problem and resend ONLY the corrected lines to append.\n"
        f"Output ONLY the RCL lines to append. No explanation, no markdown fences."
    )


def call_model(prompt: str, model: str = None, max_backoff_retries: int = 4) -> dict:
    """One-shot stateless call — still used for the CoT baseline (Part B),
    which is a single call with no retry/repair loop, so there's no
    resend-the-context problem to fix here. Kept separate from
    call_model_chat below, which is for the multi-turn generate_and_splice
    loop where the resend problem actually occurs."""
    model = model or MODEL
    delay = 2
    last_exc = None
    for attempt in range(1, max_backoff_retries + 1):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            elapsed = time.perf_counter() - t0
            text = response.text.strip()
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            return {"text": text.strip("\n"), "usage": extract_usage(response), "time_seconds": elapsed}
        except Exception as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            is_server_error = (status and 500 <= int(status) < 600) or "UNAVAILABLE" in str(e)
            last_exc = e
            if not is_server_error or attempt == max_backoff_retries:
                raise
            print(f"  [transient server error, retrying in {delay}s: {e}]")
            time.sleep(delay)
            delay *= 2
    raise last_exc


def call_model_chat(chat, message: str, max_backoff_retries: int = 4) -> dict:
    """Sends one turn on a persistent chat session (chat.send_message) instead
    of a stateless client.models.generate_content call. Same backoff logic as
    before for transient 5xx errors — that part hasn't changed, only the
    transport (session turn vs. one-shot call).

    Returns {"text": ..., "usage": {...}, "time_seconds": ...}, same shape as
    the old call_model, so every downstream accounting call site is untouched."""
    delay = 2
    last_exc = None
    for attempt in range(1, max_backoff_retries + 1):
        t0 = time.perf_counter()
        try:
            response = chat.send_message(message)
            elapsed = time.perf_counter() - t0
            text = response.text.strip()
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            return {"text": text.strip("\n"), "usage": extract_usage(response), "time_seconds": elapsed}
        except Exception as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            is_server_error = (status and 500 <= int(status) < 600) or "UNAVAILABLE" in str(e)
            last_exc = e
            if not is_server_error or attempt == max_backoff_retries:
                raise
            print(f"  [transient server error, retrying in {delay}s: {e}]")
            time.sleep(delay)
            delay *= 2
    raise last_exc


def indent_block(text: str, base_depth_spaces: int = 4, step: int = 4) -> str:
    """Grammar-driven normalization, not just whitespace guessing.
    The RCL grammar only allows REASON to nest under REASON — SOLUTION,
    DELTA, and CONSTRAINT are only ever valid as top-level blocks, never
    indented under a REASON. So rather than inferring nesting purely from
    the model's (often ambiguous) raw indentation, this tracks REASON
    nesting depth via a stack, and forces SOLUTION/DELTA/CONSTRAINT to the
    base level every time, regardless of how the model indented them."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    reason_stack = []  # raw indents of currently-open REASON levels
    out = []
    for ln in lines:
        raw = len(ln) - len(ln.lstrip(" "))
        stripped = ln.strip()
        keyword = stripped.split()[0] if stripped else ""

        if keyword == "REASON":
            if not reason_stack:
                reason_stack.append(raw)
            elif raw > reason_stack[-1]:
                reason_stack.append(raw)
            else:
                while len(reason_stack) > 1 and raw < reason_stack[-1]:
                    reason_stack.pop()
                if raw > reason_stack[-1]:
                    reason_stack.append(raw)
                else:
                    reason_stack[-1] = raw
            level = len(reason_stack) - 1  # 0 = top-level REASON, sibling to CONSTRAINT
            out.append(" " * (base_depth_spaces + level * step) + stripped)

        elif keyword in ("SOLUTION", "DELTA", "CONSTRAINT"):
            reason_stack = []  # these always sit at the base level, never nested
            out.append(" " * base_depth_spaces + stripped)

        else:
            # unrecognized line (shouldn't normally happen) — safe fallback to base level
            out.append(" " * base_depth_spaces + stripped)

    return "\n".join(out)


def strip_constraint_lines(generated: str) -> str:
    """Used in strict mode as a hard backstop: even if the model ignores the
    instruction and emits a CONSTRAINT line anyway, remove it before splicing
    rather than trusting the prompt alone."""
    kept = [ln for ln in generated.splitlines() if not ln.strip().startswith("CONSTRAINT")]
    return "\n".join(kept)


def generate_and_splice(human_block: str, mode: str, config: RCLConfig = None) -> dict:
    """Runs the generate -> splice -> validate -> repair loop over ONE
    persistent chat session. Turn 1 carries the full spec/instructions/CSV;
    every retry after that sends only the validator's error message, since
    the session already has the heavy context from turn 1 — this is the fix
    for the 'takes too much time' issue (previously each retry rebuilt and
    resent the entire prompt, CSV included, from scratch).

    Returns a dict with the outcome, PLUS real accounting: total tokens and
    total wall-clock time across every attempt (including failed/retried
    ones — a rejected attempt still cost real tokens and real time, and a
    production accounting should reflect that, not just the winning call)."""
    config = config or RCLConfig(mode=mode)
    total_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_time = 0.0
    chat = client.chats.create(model=MODEL)
    prior_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt == 1:
            message = build_initial_prompt(human_block, mode, config)
            print(f"  [attempt {attempt}/{MAX_RETRIES}] sending full context "
                  f"({len(message)} chars, incl. embedded data)...")
        else:
            message = build_retry_prompt(prior_error)
            print(f"  [attempt {attempt}/{MAX_RETRIES}] sending correction only "
                  f"({len(message)} chars) — prior full context stays in session, not resent")

        call_result = call_model_chat(chat, message)
        generated = call_result["text"]
        for k in total_usage:
            total_usage[k] += call_result["usage"].get(k, 0)
        total_time += call_result["time_seconds"]
        print(f"  [attempt {attempt}/{MAX_RETRIES}] got response in "
              f"{call_result['time_seconds']:.2f}s "
              f"({call_result['usage'].get('total_tokens', 0)} tokens this turn, "
              f"running total {total_usage['total_tokens']})")

        if mode == "strict":
            generated = strip_constraint_lines(generated)

        generated = indent_block(generated, base_depth_spaces=4, step=4)
        combined = human_block.rstrip("\n") + "\n" + generated + "\n"

        print(f"  [attempt {attempt}/{MAX_RETRIES}] validating...")
        ok, result = try_parse_and_validate(combined, config)
        if ok:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] PASSED")
            return {
                "mode": mode,
                "attempt": attempt,
                "success": True,
                "combined_source": combined,
                "problems": result,
                "total_usage": total_usage,
                "total_time_seconds": total_time,
            }
        print(f"  [attempt {attempt}/{MAX_RETRIES}] FAILED: {result}")
        prior_error = result

    print(f"  [generate_and_splice] exhausted {MAX_RETRIES} attempts, giving up")
    return {
        "mode": mode,
        "attempt": MAX_RETRIES,
        "success": False,
        "combined_source": combined,
        "last_error": prior_error,
        "total_usage": total_usage,
        "total_time_seconds": total_time,
    }

# %% [visualization]
def _node_depth(node, roots):
    d = 0
    n = node
    while n.parent is not None:
        if n in roots:
            break
        d += 1
        n = n.parent
    return d


def visualize_tree(pp, title=None, save_path=None, highlight_changed=None):
    """Renders one PROBLEM's tree: PROBLEM -> CONSTRAINT(s) -> REASON tree -> SOLUTION -> DELTA.
    Color = confidence (subtree_confidence for REASON nodes, own confidence for
    everything else) on a red-to-green scale. DECIDE_AT nodes get a gold star
    marker. UNSUPPORTED-cite nodes get a dashed border. model_added CONSTRAINTs
    get a rounded square instead of a circle, so provenance is visible at a glance.
    Dashed blue lines trace citation links (evidence trail), separate from the
    solid grey tree structure — this is the actual interpretability payoff.

    highlight_changed: optional dict from update_node()'s subtree_confidence_changes
    (node_id -> (old, new)) — draws a blue ripple ring + before/after label on
    any node whose confidence changed as a consequence of a correction, and a
    red "NEEDS REVIEW" flag above any SOLUTION marked needs_review."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    try:
        cmap = plt.colormaps.get_cmap("RdYlGn")  # Matplotlib >= 3.7
    except AttributeError:
        from matplotlib.cm import get_cmap
        cmap = get_cmap("RdYlGn")  # fallback for older Matplotlib

    highlight_changed = highlight_changed or {}

    def color_for(conf):
        if conf is None:
            return (0.7, 0.7, 0.7, 1.0)
        return cmap(max(0.0, min(1.0, conf)))

    ROW_GAP = 1.9  # vertical distance between rows — must clear node radius (0.4) +
                    # label height below (~0.7) + star/ripple annotation above (~0.55)

    positions = {}
    node_lookup = {}
    edges = []

    node_lookup[pp.problem.node_id] = pp.problem
    positions[pp.problem.node_id] = (0.0, 0.0)

    for i, c in enumerate(pp.constraints):
        x = (i - (len(pp.constraints) - 1) / 2.0) * 2.0
        positions[c.node_id] = (x, -1.0 * ROW_GAP)
        node_lookup[c.node_id] = c
        edges.append((pp.problem.node_id, c.node_id, "tree"))

    next_x = [0.0]
    max_depth = [0]

    def assign(node, depth):
        node_lookup[node.node_id] = node
        max_depth[0] = max(max_depth[0], depth)
        if not node.children:
            x = next_x[0]
            next_x[0] += 1.5
        else:
            xs = [assign(c, depth + 1) for c in node.children]
            x = sum(xs) / len(xs)
        positions[node.node_id] = (x, -(2.0 + depth) * ROW_GAP)
        return x

    for root in pp.reason_roots:
        assign(root, 0)
        edges.append((pp.problem.node_id, root.node_id, "tree"))
    for n in _all_reason_nodes(pp.reason_roots):
        for c in n.children:
            edges.append((n.node_id, c.node_id, "tree"))

    bottom_y = -(2.0 + max_depth[0] + 1.0) * ROW_GAP
    all_xs = [positions[nid][0] for nid in positions] or [0.0]
    center_x = (min(all_xs) + max(all_xs)) / 2.0

    for i, s in enumerate(pp.solutions):
        x = center_x + (i - (len(pp.solutions) - 1) / 2.0) * 2.0
        positions[s.node_id] = (x, bottom_y)
        node_lookup[s.node_id] = s

    all_reasons = _all_reason_nodes(pp.reason_roots)
    all_ids = {n.node_id for n in [pp.problem] + pp.constraints + all_reasons + pp.solutions}
    for n in all_reasons + pp.solutions:
        if n.cite and n.cite != "UNSUPPORTED":
            for nid in all_ids:
                if nid != n.node_id and re.search(rf"\b{re.escape(nid)}\b", n.cite):
                    edges.append((nid, n.node_id, "cite"))

    fig_height = max(5.0, (abs(bottom_y) + 2.5) * 0.85)
    fig, ax = plt.subplots(figsize=(max(8, next_x[0] * 1.3), fig_height))

    for a, b, style in edges:
        if a not in positions or b not in positions:
            continue
        xa, ya = positions[a]
        xb, yb = positions[b]
        if style == "tree":
            ax.plot([xa, xb], [ya, yb], color="#888888", linewidth=1.5, zorder=1)
        else:
            ax.plot([xa, xb], [ya, yb], color="#4477cc", linewidth=1.0,
                    linestyle="--", alpha=0.6, zorder=1)

    for nid, (x, y) in positions.items():
        node = node_lookup[nid]
        conf = node.subtree_confidence if node.kind == "REASON" and node.subtree_confidence is not None else node.confidence
        face = color_for(conf)
        is_decide = getattr(node, "decide_at", False)
        is_unsupported = getattr(node, "cite", None) == "UNSUPPORTED"
        is_model_added = getattr(node, "model_added", False)

        edgecolor = "#B8860B" if is_decide else "#333333"
        linewidth = 3 if is_decide else 1.2
        linestyle = "dashed" if is_unsupported else "solid"

        if node.kind == "CONSTRAINT" and is_model_added:
            patch = mpatches.FancyBboxPatch(
                (x - 0.45, y - 0.3), 0.9, 0.6, boxstyle="round,pad=0.02",
                facecolor=face, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, zorder=2,
            )
        else:
            patch = mpatches.Circle((x, y), 0.4, facecolor=face,
                                     edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, zorder=2)
        ax.add_patch(patch)

        if is_decide:
            ax.plot(x, y + 0.55, marker="*", markersize=14, color="#B8860B", zorder=3)

        if nid in highlight_changed:
            old, new = highlight_changed[nid]
            ax.add_patch(mpatches.Circle((x, y), 0.52, facecolor="none",
                                          edgecolor="#2255cc", linewidth=2.5,
                                          linestyle=(0, (3, 2)), zorder=4))
            ax.text(x, y + 0.75, f"{old:.2f} → {new:.2f}", ha="center", va="bottom",
                    fontsize=7, color="#2255cc", fontweight="bold")

        if node.kind == "SOLUTION" and getattr(node, "needs_review", False):
            ax.text(x, y + 0.62, "⚠ NEEDS REVIEW", ha="center", va="bottom",
                    fontsize=8, color="#cc2222", fontweight="bold")

        label = f"{nid}\n{node.text[:28]}{'...' if len(node.text) > 28 else ''}"
        ax.text(x, y - 0.62, label, ha="center", va="top", fontsize=7, wrap=True)
        conf_label = f"{conf:.2f}" if conf is not None else "?"
        ax.text(x, y, conf_label, ha="center", va="center", fontsize=8, fontweight="bold")

    if pp.delta_text:
        ax.text(center_x, bottom_y - 1.1, f"DELTA: {pp.delta_text[:60]}",
                ha="center", va="top", fontsize=8, style="italic", color="#444444")

    legend_elems = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#B8860B", markersize=12, label="DECIDE_AT"),
        mpatches.Patch(facecolor="none", edgecolor="#333333", linestyle="dashed", label="UNSUPPORTED cite"),
        mpatches.FancyBboxPatch((0, 0), 1, 1, facecolor="#cccccc", edgecolor="#333333", label="model_added constraint"),
        Line2D([0], [0], color="#4477cc", linestyle="--", label="citation link"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7, framealpha=0.9)

    ax.set_title(title or f"PROBLEM {pp.problem.node_id}: {pp.problem.text}", fontsize=10)
    ax.set_xlim(min(all_xs) - 1.5, max(all_xs) + 1.5)
    ax.set_ylim(bottom_y - 2.0, 1.0)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


# %% [PART A — case data: a real, deterministic dataset, not embedded tree nodes]
# ============================================================
# Design decision: 500 suspects do NOT become 500 RCL nodes. That would be
# unreadable, expensive, and would defeat the entire premise of a tree a
# human can audit at a glance. Instead: the 500 suspects live in a real
# dataset; RCL's tree narrates the NARROWING LOGIC (a handful of nodes),
# and a real (not mock) validator re-runs the actual filter against the
# actual data to confirm the model's stated counts are true, not guessed.
#
# This is entirely fictional/synthetic — invented suspect IDs, invented
# case, invented attributes. Not modeled on any real case.
# ============================================================
import random


def build_case_data(seed=42, n_suspects=500) -> list:
    random.seed(seed)
    data = [{
        "id": "S001", "alibi_verified": False, "motive_score": 9,
        "was_near_scene": True, "forensic_match": 0.95,
    }]
    for i in range(2, n_suspects + 1):
        data.append({
            "id": f"S{i:03d}",
            "alibi_verified": random.random() < 0.55,
            "motive_score": random.randint(0, 10),
            "was_near_scene": random.random() < 0.15,
            "forensic_match": round(random.uniform(0, 1), 2),
        })

    def passes_all(s):
        return (not s["alibi_verified"] and s["motive_score"] >= 7
                and s["was_near_scene"] and s["forensic_match"] >= 0.9)

    matches = [s for s in data if passes_all(s)]
    while len(matches) > 1:
        for s in matches:
            if s["id"] != "S001":
                s["forensic_match"] = 0.5
        matches = [s for s in data if passes_all(s)]
    return data


def apply_filters(data: list) -> dict:
    after_alibi = [s for s in data if not s["alibi_verified"]]
    after_motive = [s for s in after_alibi if s["motive_score"] >= 7]
    after_opportunity = [s for s in after_motive if s["was_near_scene"]]
    after_forensic = [s for s in after_opportunity if s["forensic_match"] >= 0.9]
    return {"after_alibi": after_alibi, "after_motive": after_motive,
            "after_opportunity": after_opportunity, "after_forensic": after_forensic}


def to_csv(rows: list) -> str:
    lines = ["id,alibi_verified,motive_score,was_near_scene,forensic_match"]
    for s in rows:
        lines.append(f"{s['id']},{s['alibi_verified']},{s['motive_score']},{s['was_near_scene']},{s['forensic_match']}")
    return "\n".join(lines)


CASE_DATA = build_case_data(n_suspects=NUM_SUSPECTS)
GT = apply_filters(CASE_DATA)
print(f"Case data generated (seed=42, n={NUM_SUSPECTS}). Ground-truth narrowing funnel:")
print(f"  {NUM_SUSPECTS} -> after alibi filter: {len(GT['after_alibi'])} "
      f"-> after motive filter: {len(GT['after_motive'])} "
      f"-> after opportunity filter: {len(GT['after_opportunity'])} "
      f"-> after forensic filter: {len(GT['after_forensic'])} "
      f"({[s['id'] for s in GT['after_forensic']]})")
TRUE_SUSPECT = GT["after_forensic"][0]["id"]


def check_crime_answer(text: str) -> dict:
    """Checklist against the REAL recomputed filter results — not a fuzzy
    text-similarity check. Each count either matches the true funnel or it
    doesn't; this is what a real grounding validator would also check."""
    numbers = [int(n) for n in re.findall(r"\b\d+\b", text)]

    def has(n):
        return n in numbers

    items = {
        "alibi_count": has(len(GT["after_alibi"])),
        "motive_count": has(len(GT["after_motive"])),
        "opportunity_count": has(len(GT["after_opportunity"])),
        "forensic_count": has(len(GT["after_forensic"])),
        "correct_suspect_id": TRUE_SUSPECT in text,
    }
    items["score"] = sum(1 for k, v in items.items() if k != "score" and v)
    items["total"] = 5
    return items


def crime_grounding_validator(node) -> ValidatorResult:
    """Real grounding check: re-runs the actual filters against the actual
    dataset and checks the SOLUTION's stated numbers against them. This is
    the RAG-pairing extension point made concrete with genuine retrieval,
    not a small fixed mock knowledge base."""
    result = check_crime_answer(node.text)
    if result["score"] == result["total"]:
        return ValidatorResult(True, "all counts and final suspect ID match the real dataset")
    failed = [k for k, v in result.items() if k not in ("score", "total") and not v]
    return ValidatorResult(False, f"{result['score']}/{result['total']} correct — failed: {failed}")


# %% [PART B — CoT baseline: single pass over the full 500-row dataset]
print("\n" + "=" * 70)
print("PART B — Chain-of-Thought baseline (single pass, no correction loop)")
print("=" * 70)

full_csv = to_csv(CASE_DATA)
cot_prompt = f"""You are investigating a case with {NUM_SUSPECTS} suspects. Apply these filters IN ORDER
and report the count remaining after each step, then name the final suspect ID:

1. Exclude any suspect with alibi_verified = True.
2. Among those remaining, keep only motive_score >= 7.
3. Among those remaining, keep only was_near_scene = True.
4. Among those remaining, keep only forensic_match >= 0.9.

Suspect data (CSV):
{full_csv}

Report all four counts and the final suspect ID clearly."""

cot_call = call_model(cot_prompt)
cot_check = check_crime_answer(cot_call["text"])
print(f"tokens={cot_call['usage']['total_tokens']}  time={cot_call['time_seconds']:.2f}s  "
      f"score={cot_check['score']}/{cot_check['total']}")
print(f"failed items: {[k for k, v in cot_check.items() if k not in ('score','total') and not v]}")
print(cot_call["text"][:1000])


# %% [PART C — RCL: two LINKED problems, cross-problem citation, real grounding validator]
print("\n" + "=" * 70)
print("PART C — RCL: linked sub-investigations with real grounding")
print("=" * 70)

CRIME_CONFIG = RCLConfig(
    min_reasons=1, max_tree_depth=4, min_confidence_threshold=0.6,
    require_constraint_coverage=True,
    solution_validators=[("crime_grounding", crime_grounding_validator)],
    validator_mode="strict",  # strict here: a wrong count should be REJECTED and retried,
                               # not just noted — this is the actual mechanism behind any
                               # accuracy edge over CoT, not a smarter model
)

# --- Sub-investigation 1: alibi + motive narrowing over the full 500 ---
P1_INPUT = f'''PROBLEM p1 [confidence: 0.7]: "Case file: {NUM_SUSPECTS} suspects. Narrow the pool using alibi verification and motive score. Full suspect data (CSV): {full_csv}"
    CONSTRAINT c1 [confidence: 0.95] [cite: "case protocol"]: "Exclude any suspect with alibi_verified = True before applying any other filter"
    CONSTRAINT c2 [confidence: 0.9] [cite: "case protocol"]: "Among remaining suspects, keep only those with motive_score >= 7"
'''
total_tokens = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
total_time = 0.0

p1_result = generate_and_splice(P1_INPUT, "strict", CRIME_CONFIG)
for k in total_tokens:
    total_tokens[k] += p1_result["total_usage"][k]
total_time += p1_result["total_time_seconds"]
print(f"\n--- p1 (alibi + motive narrowing) ---")
print(f"success={p1_result['success']}  attempts={p1_result['attempt']}  "
      f"tokens={p1_result['total_usage']['total_tokens']}  time={p1_result['total_time_seconds']:.2f}s")
if not p1_result["success"]:
    print("FINAL ERROR:", p1_result["last_error"])
print(p1_result["combined_source"])

# --- Sub-investigation 2: builds on p1's conclusion, cites it by node id ---
remaining_90_csv = to_csv(GT["after_motive"])
P2_INPUT = f'''PROBLEM p2 [confidence: 0.7]: "Building on the {len(GT['after_motive'])} candidates identified in sol_narrow1, narrow further by opportunity and forensic match to identify the suspect. Candidate data (CSV): {remaining_90_csv}"
    CONSTRAINT c3 [confidence: 0.95] [cite: "sol_narrow1"]: "Must build only on the candidates already identified in sol_narrow1, not the original {NUM_SUSPECTS}"
    CONSTRAINT c4 [confidence: 0.9] [cite: "case protocol"]: "Among remaining, keep only suspects with was_near_scene = True"
    CONSTRAINT c5 [confidence: 0.9] [cite: "case protocol"]: "Among remaining, keep only suspects with forensic_match >= 0.9"
'''
p2_result = generate_and_splice(P2_INPUT, "strict", CRIME_CONFIG)
for k in total_tokens:
    total_tokens[k] += p2_result["total_usage"][k]
total_time += p2_result["total_time_seconds"]
print(f"\n--- p2 (opportunity + forensic narrowing, cites p1) ---")
print(f"success={p2_result['success']}  attempts={p2_result['attempt']}  "
      f"tokens={p2_result['total_usage']['total_tokens']}  time={p2_result['total_time_seconds']:.2f}s")
if not p2_result["success"]:
    print("FINAL ERROR:", p2_result["last_error"])
print(p2_result["combined_source"])

rcl_check = {"score": 0, "total": 5}
if p1_result["success"] and p2_result["success"]:
    for pp in p2_result["problems"]:
        for sol in pp.solutions:
            rcl_check = check_crime_answer(sol.text)
            print(f"  SOLUTION {sol.node_id} score={rcl_check['score']}/{rcl_check['total']}")

print(f"\nRCL total (both problems): tokens={total_tokens['total_tokens']}  time={total_time:.2f}s")

# --- comparison chart ---
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
colors = ["#cc8844", "#4477cc"]
approaches = ["Chain-of-Thought", "RCL (linked, strict)"]
tok_vals = [cot_call["usage"]["total_tokens"], total_tokens["total_tokens"]]
time_vals = [cot_call["time_seconds"], total_time]
score_vals = [cot_check["score"], rcl_check["score"]]

axes[0].bar(approaches, tok_vals, color=colors)
axes[0].set_title("Total tokens")
for i, v in enumerate(tok_vals):
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
axes[1].bar(approaches, time_vals, color=colors)
axes[1].set_title("Wall-clock time (s)")
for i, v in enumerate(time_vals):
    axes[1].text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
axes[2].bar(approaches, score_vals, color=colors)
axes[2].set_title("Checklist score (out of 5)")
axes[2].set_ylim(0, 5.5)
for i, v in enumerate(score_vals):
    axes[2].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
fig.suptitle("RCL (linked, grounded) vs. Chain-of-Thought — 500-suspect narrowing", fontsize=11)
plt.tight_layout()
crime_chart_path = OUT_DIR + "/rcl_vs_cot_crime_comparison.png"
plt.savefig(crime_chart_path, dpi=150, bbox_inches="tight")
plt.show()
download_if_colab(crime_chart_path)

print("\nNote: if RCL scores higher, the mechanism is the validator_mode='strict' "
      "grounding check rejecting a wrong count and forcing a retry — not a smarter "
      "model. If CoT also scores well here, report that honestly; the real point is "
      "that RCL's correctness is MECHANICALLY enforced, not just observed once.")


# %% [PART D — visualize both linked trees + cross-problem citation]
if p1_result.get("success"):
    for pp in p1_result["problems"]:
        path = OUT_DIR + "/crime_p1.png"
        visualize_tree(pp, title="p1 — alibi + motive narrowing", save_path=path)
        download_if_colab(path)

if p2_result.get("success"):
    for pp in p2_result["problems"]:
        path = OUT_DIR + "/crime_p2.png"
        visualize_tree(pp, title="p2 — opportunity + forensic narrowing (cites p1)", save_path=path)
        download_if_colab(path)


# %% [PART E — the twist: new evidence resurfaces post-arrest]
# ============================================================
# This is the actual real-world-impact payoff: after a suspect is
# identified (and, narratively, arrested), a piece of forensic evidence is
# later found faulty. A real audit trail should flag EXACTLY which
# downstream conclusion depended on that evidence — automatically — not
# require someone to manually re-read the whole case file. This is the
# same update_node() mechanism from the showcase demo, extended to check
# for impact ACROSS linked problems, not just within one tree.
# ============================================================
def update_node_cross_problem(all_pps: list, node_id: str, **kwargs) -> dict:
    """Like update_node(), but flags impacted SOLUTIONs across every linked
    problem, not just the one containing the corrected node — necessary
    once reasoning is split across multiple cited PROBLEM blocks."""
    owner = None
    for pp in all_pps:
        if find_node(pp, node_id) is not None:
            owner = pp
            break
    if owner is None:
        raise RCLError(f"update_node_cross_problem: node '{node_id}' not found in any linked problem")

    report = update_node(owner, node_id, **kwargs)

    node = find_node(owner, node_id)
    affected_ids = {node_id}
    if node.kind == "REASON":
        affected_ids |= {n.node_id for n in _all_reason_nodes([node])}
    affected_ids |= set(report["flagged_solutions"])

    cross_flagged = []
    for pp in all_pps:
        if pp is owner:
            continue
        for sol in pp.solutions:
            if sol.cite and any(re.search(rf"\b{re.escape(aid)}\b", sol.cite) for aid in affected_ids):
                sol.needs_review = True
                cross_flagged.append(f"{pp.problem.node_id}.{sol.node_id}")

    report["cross_problem_flagged"] = cross_flagged
    return report


print("\n" + "=" * 70)
print("PART E — post-arrest: new forensic evidence resurfaces")
print("=" * 70)

if p1_result.get("success") and p2_result.get("success"):
    all_pps = p1_result["problems"] + p2_result["problems"]

    p2_tree = p2_result["problems"][0]
    forensic_nodes = [n for n in _all_reason_nodes(p2_tree.reason_roots)
                       if "forensic" in n.text.lower() or n.decide_at]
    target = forensic_nodes[0] if forensic_nodes else p2_tree.reason_roots[0]

    print(f"Simulating: the forensic lab retracts its match report. {target.node_id}'s "
          f"claim is no longer supported.")
    report = update_node_cross_problem(
        all_pps, target.node_id, new_confidence=0.1, new_cite="UNSUPPORTED",
        update_reason="Forensic lab retracted the match report post-arrest; original analysis found to be contaminated.",
    )
    print("Backprop report:", report)
    if report["flagged_solutions"]:
        print(f"SOLUTION(s) flagged within p2: {report['flagged_solutions']}")
    if report["cross_problem_flagged"]:
        print(f"SOLUTION(s) flagged in OTHER linked problems: {report['cross_problem_flagged']}")
    else:
        print("No other linked problem's SOLUTION depended on this node — "
              "p1's conclusion (alibi + motive narrowing) is correctly untouched, "
              "since it never relied on the forensic evidence.")

    path = OUT_DIR + "/crime_p2_after_correction.png"
    visualize_tree(p2_tree, title="p2 AFTER forensic retraction (ripple highlighted)",
                    highlight_changed=report["subtree_confidence_changes"], save_path=path)
    download_if_colab(path)
else:
    print("p1 or p2 did not succeed above — nothing to run the correction scene on.")
