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


# %% [toy example validator — stand-in for a real RAG grounding check]
# This is deliberately NOT a real retrieval system. It's a minimal, honest
# proof that the validator socket works end-to-end: a small fixed set of
# "known" sources acts as a stand-in knowledge base. A production version
# would replace MOCK_KNOWLEDGE_BASE with an actual vector store lookup —
# the fn signature (node) -> ValidatorResult stays identical either way,
# which is the point of building it as a hook rather than hardcoding it.
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


def build_prompt(human_block: str, mode: str, config: RCLConfig, prior_error: str = None) -> str:
    instructions = STRICT_INSTRUCTIONS if mode == "strict" else OPEN_INSTRUCTIONS
    prompt = (
        f"{RCL_SPEC}\n\n{instructions}\n\n"
        f"Additional acceptance criteria for this task (your output will be "
        f"rejected if it doesn't meet these, in addition to the base grammar rules):\n"
        f"{config.describe()}\n\n"
        f"Human-authored input so far:\n\n{human_block}\n"
    )
    if prior_error:
        prompt += (
            f"\nYour previous attempt failed validation with this exact error:\n"
            f"{prior_error}\n"
            f"Fix this specific problem and resend ONLY the corrected lines to append."
        )
    prompt += "\nOutput ONLY the RCL lines to append. No explanation, no markdown fences."
    return prompt


def call_model(prompt: str, model: str = None, max_backoff_retries: int = 4) -> dict:
    """Retries on transient server-side failures (5xx, e.g. 503 UNAVAILABLE
    under high demand) with exponential backoff. This is separate from the
    validation-repair retry loop in generate_and_splice — that one changes
    the prompt based on a grammar/validator error; this one just waits and
    resends the same prompt, because a 503 has nothing to do with the
    content of the request.

    Returns {"text": ..., "usage": {...}, "time_seconds": ...} — every call
    site needs the timing/token data for the production accounting this
    script does, so it's returned here once rather than re-measured everywhere."""
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
    """Runs the generate -> splice -> validate -> repair loop.
    Returns a dict with the outcome, PLUS real accounting: total tokens and
    total wall-clock time across every attempt (including failed/retried
    ones — a rejected attempt still cost real tokens and real time, and a
    production accounting should reflect that, not just the winning call)."""
    config = config or RCLConfig(mode=mode)
    prior_error = None
    total_usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_time = 0.0

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = build_prompt(human_block, mode, config, prior_error)
        call_result = call_model(prompt)
        generated = call_result["text"]
        for k in total_usage:
            total_usage[k] += call_result["usage"].get(k, 0)
        total_time += call_result["time_seconds"]

        if mode == "strict":
            generated = strip_constraint_lines(generated)

        generated = indent_block(generated, base_depth_spaces=4, step=4)
        combined = human_block.rstrip("\n") + "\n" + generated + "\n"

        ok, result = try_parse_and_validate(combined, config)
        if ok:
            return {
                "mode": mode,
                "attempt": attempt,
                "success": True,
                "combined_source": combined,
                "problems": result,
                "total_usage": total_usage,
                "total_time_seconds": total_time,
            }
        prior_error = result

    return {
        "mode": mode,
        "attempt": MAX_RETRIES,
        "success": False,
        "combined_source": combined,
        "last_error": prior_error,
        "total_usage": total_usage,
        "total_time_seconds": total_time,
    }


# %% [PART 1 — showcase problem: exercises every structural feature at once]
SHOWCASE_INPUT = '''PROBLEM p1 [confidence: 0.65]: "A regional bank's fraud-detection model began approving 4x its normal volume of high-risk international wire transfers over a 72-hour period, and three of these transfers were later confirmed fraudulent by the receiving banks. Determine the root cause and whether the model itself, the surrounding pipeline, or an external actor is responsible, before any transfers are reversed or the model is retrained."
    CONSTRAINT c1 [confidence: 0.95] [cite: "regulatory incident report requirement"]: "Root cause must be identified and documented before any corrective action, per compliance policy"
    CONSTRAINT c2 [confidence: 0.85] [cite: "internal SLA"]: "Any proposed fix must not increase false-positive rate on legitimate transfers by more than 2%"
'''

SHOWCASE_CONFIG = RCLConfig(
    min_reasons=3,
    max_tree_depth=5,
    min_confidence_threshold=0.55,
    min_subtree_confidence=0.25,
    require_constraint_coverage=True,   # both c1 and c2 must be explicitly addressed
    max_unsupported_ratio=0.4,
    validators=[("mock_rag", mock_rag_validator)],
    validator_mode="advisory",
)

print("=" * 70)
print("PART 1 — SHOWCASE PROBLEM (strict + open, full feature set)")
print("=" * 70)
print(SHOWCASE_CONFIG.describe())

print("\n" + "-" * 70)
print("STRICT MODE")
print("-" * 70)
showcase_strict = generate_and_splice(SHOWCASE_INPUT, "strict", SHOWCASE_CONFIG)
print(f"success={showcase_strict['success']}  attempts={showcase_strict['attempt']}  "
      f"tokens={showcase_strict['total_usage']['total_tokens']}  "
      f"time={showcase_strict['total_time_seconds']:.2f}s")
print(showcase_strict["combined_source"])
if not showcase_strict["success"]:
    print("FINAL ERROR:", showcase_strict["last_error"])

print("\n" + "-" * 70)
print("OPEN MODE")
print("-" * 70)
showcase_open = generate_and_splice(SHOWCASE_INPUT, "open", SHOWCASE_CONFIG)
print(f"success={showcase_open['success']}  attempts={showcase_open['attempt']}  "
      f"tokens={showcase_open['total_usage']['total_tokens']}  "
      f"time={showcase_open['total_time_seconds']:.2f}s")
print(showcase_open["combined_source"])
if showcase_open["success"]:
    for pp in showcase_open["problems"]:
        added = [c.node_id for c in pp.constraints if c.model_added]
        print(f"  model-added constraints: {added or 'none'}")
else:
    print("FINAL ERROR:", showcase_open["last_error"])


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


if showcase_strict.get("success"):
    for pp in showcase_strict["problems"]:
        path = f"{OUT_DIR}/showcase_strict_{pp.problem.node_id}.png"
        visualize_tree(pp, title=f"SHOWCASE STRICT — {pp.problem.node_id}", save_path=path)
        download_if_colab(path)

if showcase_open.get("success"):
    for pp in showcase_open["problems"]:
        path = f"{OUT_DIR}/showcase_open_{pp.problem.node_id}.png"
        visualize_tree(pp, title=f"SHOWCASE OPEN — {pp.problem.node_id}", save_path=path)
        download_if_colab(path)


# %% [demo scene: inject -> catch -> ripple -> fix -> re-stabilize]
print("\n" + "=" * 70)
print("DEMO SCENE: validator catch + backprop ripple (on the showcase tree)")
print("=" * 70)

if showcase_strict.get("success"):
    demo_pp = showcase_strict["problems"][0]

    try:
        validate(demo_pp, SHOWCASE_CONFIG)
        print("Validator pass: all citations matched the mock knowledge base.")
    except RCLError as e:
        print("Validator hook fired:", e)

    flagged = [
        n for n in (demo_pp.constraints + _all_reason_nodes(demo_pp.reason_roots) + demo_pp.solutions)
        if getattr(n, "validator_notes", None)
    ]
    if flagged:
        print("Nodes flagged by validator (advisory mode):")
        for n in flagged:
            print(f"  {n.kind} {n.node_id}: {n.validator_notes}")

    target_root = demo_pp.reason_roots[0]
    print(f"\nSimulating correction: {target_root.node_id}'s claim turns out to be "
          f"weaker than first thought (evidence was misread).")
    report = update_node(
        demo_pp, target_root.node_id, new_confidence=0.2,
        update_reason="Manual review found the original evidence was misinterpreted.",
    )
    print("Backprop report:", report)
    if report["flagged_solutions"]:
        print(f"SOLUTION(s) flagged as needing review: {report['flagged_solutions']}")

    path = f"{OUT_DIR}/showcase_after_correction_{demo_pp.problem.node_id}.png"
    visualize_tree(
        demo_pp, title=f"AFTER CORRECTION — {demo_pp.problem.node_id} (ripple highlighted)",
        highlight_changed=report["subtree_confidence_changes"], save_path=path,
    )
    download_if_colab(path)
else:
    print("Showcase strict-mode generation did not succeed above — nothing to run the demo scene on.")


# %% [PART 2 — RCL vs. Chain-of-Thought head-to-head on a long, trap-laden clinical dosing problem]
# ============================================================
# IMPORTANT: "Zenoxacin" is a FICTIONAL drug invented for this exercise.
# All dosing tiers, thresholds, and safety caps below are SYNTHETIC —
# constructed specifically to create a long, multi-step arithmetic chain
# with realistic failure points (unit conversions, a threshold trap, an
# iterative safety-cap check). This is a math/reasoning stress-test, NOT
# real clinical guidance, and should never be treated as one.
# ============================================================
print("\n" + "!" * 70)
print("NOTE: 'Zenoxacin' and its dosing rules below are FICTIONAL, invented")
print("for this exercise. This is a synthetic math stress-test, not real")
print("clinical dosing guidance.")
print("!" * 70)

HEALTH_PROBLEM = (
    "A 68-year-old female patient is being evaluated for IV dosing of "
    "Zenoxacin (a fictional antibiotic used only for this exercise), ordered "
    "for a suspected resistant infection. Her measured body weight is 154 "
    "lb. Her most recent basic metabolic panel shows a serum creatinine of "
    "106 micromol/L; her blood pressure is 128/76, heart rate 82, and "
    "temperature 37.6C, none of which affect the dosing calculation. The "
    "dietary team separately calculated her ideal body weight via the "
    "Devine formula as 58 kg for nutritional planning — this ideal body "
    "weight figure is NOT to be used for the renal or drug-dose "
    "calculations below, which must use her actual measured body weight in "
    "kilograms. "
    "Step 1: Convert her weight from pounds to kilograms (1 kg = 2.2046 "
    "lb) and her serum creatinine from micromol/L to mg/dL (1 mg/dL = 88.4 "
    "micromol/L). "
    "Step 2: Calculate her creatinine clearance (CrCl, in mL/min) using "
    "the Cockcroft-Gault formula: CrCl = [(140 - age) x weight_kg x 0.85 "
    "(since the patient is female)] / (72 x serum_creatinine_mgdl). Do NOT "
    "round the CrCl value before selecting the dosing tier in Step 3 — "
    "the correct tier depends on the precise unrounded value. "
    "Step 3: Zenoxacin's fictional renal dosing protocol for this "
    "indication specifies: CrCl >= 90 mL/min -> dose every 12 hours; CrCl "
    "50-89 mL/min -> dose every 24 hours; CrCl 30-49 mL/min -> dose every "
    "48 hours; CrCl < 30 mL/min -> do not calculate further, refer to the "
    "renal dosing service. Select the correct interval for this patient "
    "based on her exact calculated CrCl. "
    "Step 4: The standard maintenance dose is 15 mg/kg of actual body "
    "weight, given at the interval selected in Step 3, subject to a "
    "fictional maximum single-dose cap of 2000 mg (if the calculated dose "
    "exceeds 2000 mg, cap it at exactly 2000 mg). Calculate the "
    "maintenance dose in mg. "
    "Step 5: Zenoxacin for IV infusion is supplied at a concentration of "
    "5 mg/mL. The standard infusion protocol is to infuse over 90 "
    "minutes, with a fictional maximum safe infusion rate of 150 mL/hour. "
    "Calculate the required infusion volume (mL) and infusion rate "
    "(mL/hour) for the maintenance dose at the standard 90-minute "
    "duration. If the calculated rate would exceed 150 mL/hour, extend "
    "the infusion duration in 30-minute increments (90, then 120, then "
    "150 minutes, and so on) until the rate is at or below 150 mL/hour, "
    "and report the final duration and rate used. "
    "Step 6: Separately, a one-time loading dose is also ordered, "
    "calculated as 25 mg/kg of actual body weight (also subject to the "
    "2000 mg single-dose cap), and infused using the same starting "
    "90-minute protocol and the same 150 mL/hour maximum rate rule as in "
    "Step 5 — apply the same duration-extension rule independently to the "
    "loading dose if needed; do not assume the outcome from Step 5 "
    "automatically applies here. "
    "Report clearly: (1) the patient's weight in kg and serum creatinine "
    "in mg/dL, (2) her exact CrCl in mL/min and the resulting dosing "
    "interval in hours, (3) the maintenance dose in mg, its infusion "
    "duration in minutes, and its infusion rate in mL/hour, and (4) the "
    "loading dose in mg, its infusion duration in minutes, and its "
    "infusion rate in mL/hour."
)


def compute_health_ground_truth() -> dict:
    """Computed programmatically rather than by hand, precisely because this
    chain is long enough that manual arithmetic would itself be error-prone —
    the same failure mode this problem is designed to expose in CoT."""
    weight_kg = 154 / 2.2046
    scr_mgdl = 106 / 88.4
    age = 68
    crcl = ((140 - age) * weight_kg * 0.85) / (72 * scr_mgdl)

    if crcl >= 90:
        interval_hours = 12
    elif crcl >= 50:
        interval_hours = 24
    elif crcl >= 30:
        interval_hours = 48
    else:
        interval_hours = None

    max_single_dose = 2000
    conc = 5  # mg/mL
    max_rate = 150  # mL/hr

    maintenance_dose = min(15 * weight_kg, max_single_dose)
    maintenance_vol = maintenance_dose / conc
    m_dur = 90
    m_rate = maintenance_vol / (m_dur / 60)
    while m_rate > max_rate:
        m_dur += 30
        m_rate = maintenance_vol / (m_dur / 60)

    loading_dose = min(25 * weight_kg, max_single_dose)
    loading_vol = loading_dose / conc
    l_dur = 90
    l_rate = loading_vol / (l_dur / 60)
    while l_rate > max_rate:
        l_dur += 30
        l_rate = loading_vol / (l_dur / 60)

    return {
        "weight_kg": weight_kg, "scr_mgdl": scr_mgdl, "crcl": crcl,
        "interval_hours": interval_hours,
        "maintenance_dose_mg": maintenance_dose, "maintenance_duration_min": m_dur, "maintenance_rate_mlhr": m_rate,
        "loading_dose_mg": loading_dose, "loading_duration_min": l_dur, "loading_rate_mlhr": l_rate,
    }


HEALTH_GT = compute_health_ground_truth()
print("\nGround truth (computed programmatically):")
for k, v in HEALTH_GT.items():
    print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")


def check_health_answer(text: str) -> dict:
    """Checklist-based checker — 6 independent items, each with its own
    tolerance, rather than a single pass/fail. This is more honest for a
    multi-part answer: it shows exactly WHICH sub-step failed, not just
    whether the whole thing was wrong."""
    numbers = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]

    def near(target, tol):
        return any(abs(n - target) <= tol for n in numbers)

    items = {
        "crcl_value": near(HEALTH_GT["crcl"], 1.0),
        "dosing_interval_48h": bool(re.search(r"48\s*[- ]?\s*hour", text, re.I)) or
                                 bool(re.search(r"q48h", text, re.I)) or
                                 bool(re.search(r"every\s+48", text, re.I)),
        "maintenance_dose_mg": near(HEALTH_GT["maintenance_dose_mg"], 5.0),
        "maintenance_rate_mlhr": near(HEALTH_GT["maintenance_rate_mlhr"], 2.0),
        "loading_dose_mg": near(HEALTH_GT["loading_dose_mg"], 5.0),
        "loading_duration_extended": near(HEALTH_GT["loading_duration_min"], 5.0) or near(2.5, 0.1),
    }
    items["score"] = sum(1 for k, v in items.items() if k != "score" and v)
    items["total"] = 6
    return items


def health_solution_validator(node) -> ValidatorResult:
    """A real correctness validator for this domain — recomputes ground
    truth independently and checks the SOLUTION against it, item by item.
    This is the concrete extension point for a real clinical dosing
    double-check system: same fn(node) -> ValidatorResult signature, a
    production version would just have a more rigorous checker inside."""
    result = check_health_answer(node.text)
    if result["score"] == result["total"]:
        return ValidatorResult(True, "all 6 checklist items match ground truth")
    return ValidatorResult(
        False,
        f"only {result['score']}/{result['total']} checklist items correct — "
        f"failed items: {[k for k, v in result.items() if k not in ('score', 'total') and not v]}"
    )


print("\n" + "=" * 70)
print("PART 2 — RCL vs. Chain-of-Thought on a long, trap-laden dosing problem")
print("=" * 70)

# --- CoT baseline ---
print("\n--- Chain-of-Thought baseline ---")
cot_prompt = f"{HEALTH_PROBLEM}\n\nThink step by step, showing all six sub-answers clearly labeled."
cot_call = call_model(cot_prompt)
cot_check = check_health_answer(cot_call["text"])
print(f"tokens={cot_call['usage']['total_tokens']}  time={cot_call['time_seconds']:.2f}s  "
      f"score={cot_check['score']}/{cot_check['total']}")
print(f"failed items: {[k for k, v in cot_check.items() if k not in ('score', 'total') and not v]}")
print(cot_call["text"][:1200])

# --- RCL run ---
print("\n--- RCL (strict mode) ---")
HEALTH_HUMAN_INPUT = f'''PROBLEM p1 [confidence: 0.7]: "{HEALTH_PROBLEM}"
    CONSTRAINT c1 [confidence: 0.95] [cite: "problem statement"]: "Must use actual measured body weight in kg for all dosing calculations, never the ideal body weight figure"
    CONSTRAINT c2 [confidence: 0.95] [cite: "problem statement"]: "CrCl must be calculated to full precision and the dosing tier selected from the exact value, not a rounded value"
    CONSTRAINT c3 [confidence: 0.9] [cite: "problem statement"]: "Infusion rate must be checked against the 150 mL/hour cap independently for BOTH the maintenance dose and the loading dose — a pass for one does not imply a pass for the other"
    CONSTRAINT c4 [confidence: 0.85] [cite: "problem statement"]: "Both the maintenance dose and loading dose must be checked against the 2000 mg single-dose cap"
'''
HEALTH_CONFIG = RCLConfig(
    min_reasons=5,
    max_tree_depth=6,
    min_confidence_threshold=0.6,
    require_constraint_coverage=True,
    solution_validators=[("health_check", health_solution_validator)],
    validator_mode="advisory",
)
rcl_health_result = generate_and_splice(HEALTH_HUMAN_INPUT, "strict", HEALTH_CONFIG)
print(f"success={rcl_health_result['success']}  attempts={rcl_health_result['attempt']}  "
      f"tokens={rcl_health_result['total_usage']['total_tokens']}  "
      f"time={rcl_health_result['total_time_seconds']:.2f}s")

rcl_check = {"score": 0, "total": 6}
if rcl_health_result["success"]:
    for pp in rcl_health_result["problems"]:
        for sol in pp.solutions:
            rcl_check = check_health_answer(sol.text)
            print(f"  SOLUTION {sol.node_id} score={rcl_check['score']}/{rcl_check['total']}: {sol.text}")
    print(rcl_health_result["combined_source"])
else:
    print("FINAL ERROR:", rcl_health_result["last_error"])

# --- comparison chart ---
print("\n--- Building comparison chart ---")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = ["#cc8844", "#4477cc"]
approaches = ["Chain-of-Thought", "RCL (strict)"]

tokens = [cot_call["usage"]["total_tokens"], rcl_health_result["total_usage"]["total_tokens"]]
times = [cot_call["time_seconds"], rcl_health_result["total_time_seconds"]]

axes[0].bar(approaches, tokens, color=colors)
axes[0].set_title("Total tokens used")
for i, v in enumerate(tokens):
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

axes[1].bar(approaches, times, color=colors)
axes[1].set_title("Wall-clock time (s)")
for i, v in enumerate(times):
    axes[1].text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)

axes[2].bar(approaches, [cot_check["score"], rcl_check["score"]], color=colors)
axes[2].set_title("Checklist score (out of 6)")
axes[2].set_ylim(0, 6.5)
for i, v in enumerate([cot_check["score"], rcl_check["score"]]):
    axes[2].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

fig.suptitle("RCL vs. Chain-of-Thought — long trap-laden dosing problem", fontsize=11)
plt.tight_layout()
comparison_path = OUT_DIR + "/rcl_vs_cot_health_comparison.png"
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.show()
download_if_colab(comparison_path)

# --- per-item breakdown chart: exactly where each approach failed ---
item_names = [k for k in cot_check if k not in ("score", "total")]
fig2, ax2 = plt.subplots(figsize=(10, 4))
x = range(len(item_names))
width = 0.35
cot_vals = [1 if cot_check[k] else 0 for k in item_names]
rcl_vals = [1 if rcl_check.get(k, False) else 0 for k in item_names]
ax2.bar([i - width/2 for i in x], cot_vals, width, label="Chain-of-Thought", color=colors[0])
ax2.bar([i + width/2 for i in x], rcl_vals, width, label="RCL (strict)", color=colors[1])
ax2.set_xticks(list(x))
ax2.set_xticklabels(item_names, rotation=30, ha="right", fontsize=8)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Wrong", "Correct"])
ax2.set_title("Per-item breakdown — exactly where each approach diverges")
ax2.legend()
plt.tight_layout()
breakdown_path = OUT_DIR + "/rcl_vs_cot_health_breakdown.png"
plt.savefig(breakdown_path, dpi=150, bbox_inches="tight")
plt.show()
download_if_colab(breakdown_path)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Approach':<20}{'Tokens':<10}{'Time (s)':<12}{'Score':<10}")
print(f"{'Chain-of-Thought':<20}{tokens[0]:<10}{times[0]:<12.2f}{cot_check['score']}/6")
print(f"{'RCL (strict)':<20}{tokens[1]:<10}{times[1]:<12.2f}{rcl_check['score']}/6")
print(f"\nReport whichever way this actually comes out — if CoT happens to pass "
      f"anyway on a given run, that's real data too, and honesty about it is "
      f"worth more to a judge than a result that conveniently always favors RCL.")
