"""
RCL — Reasoning-Constraint Language
Parser + tree-builder + validator.

Requires: pip install lark --break-system-packages
(This sandbox has no network access, so this file is written but not yet
run here — test it in your own environment with the command above.)
"""

import re
from lark import Lark
from lark.indenter import Indenter
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Indentation handling
# ---------------------------------------------------------------------------

class RCLIndenter(Indenter):
    NL_type = "_NL"
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


def build_lark_parser(grammar_path: str = None) -> Lark:
    if grammar_path is None:
        grammar_path = Path(__file__).parent / "rcl_grammar.lark"
    grammar_text = Path(grammar_path).read_text()
    return Lark(grammar_text, parser="lalr", postlex=RCLIndenter())


# ---------------------------------------------------------------------------
# 2. Node objects — what the raw Lark parse tree gets converted into
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, kind, node_id, text, decide_at=False,
                 confidence=None, cite=None, parent=None):
        self.kind = kind              # "PROBLEM" | "CONSTRAINT" | "REASON" | "SOLUTION"
        self.node_id = node_id
        self.text = text
        self.decide_at = decide_at
        self.confidence = confidence  # declared, float or None
        self.cite = cite              # string, "UNSUPPORTED", or None
        self.parent = parent
        self.children = []
        self.subtree_confidence = None   # filled in by validator, computed not declared

    def __repr__(self):
        return f"<{self.kind} {self.node_id} conf={self.confidence} subtree={self.subtree_confidence}>"


class ParsedProblem:
    def __init__(self, problem_node, constraints, reason_roots, solutions, delta_text):
        self.problem = problem_node
        self.constraints = constraints      # list[Node]
        self.reason_roots = reason_roots     # list[Node] (top-level REASONs; each may have children)
        self.solutions = solutions          # list[Node]
        self.delta_text = delta_text        # str or None


# ---------------------------------------------------------------------------
# 3. Tree-builder — walks the Lark parse tree into Node objects
# ---------------------------------------------------------------------------

def _unquote(tok):
    return str(tok)[1:-1]  # strip the surrounding "  "


def _extract_tags(tags_tree):
    """tags_tree is a `tags` or `prob_tags` node; returns (decide_at, confidence, cite).
    Raises RCLError if the same tag type appears more than once on one node —
    silently taking the last value would be a data-loss bug, not a feature."""
    decide_at = False
    confidence = None
    cite = None
    seen = set()
    for tag in tags_tree.children:
        inner = tag.children[0]
        if inner.data in seen:
            raise RCLError(f"duplicate [{inner.data}] tag on the same node.")
        seen.add(inner.data)
        if inner.data == "decide_at":
            decide_at = True
        elif inner.data == "confidence":
            confidence = float(inner.children[0])
        elif inner.data == "cite":
            val = inner.children[0]
            cite = "UNSUPPORTED" if str(val) == "UNSUPPORTED" else _unquote(val)
    return decide_at, confidence, cite


def _build_reason(tree, parent):
    node_id = str(tree.children[0])
    decide_at, confidence, cite = _extract_tags(tree.children[1])
    text = _unquote(tree.children[2])
    node = Node("REASON", node_id, text, decide_at, confidence, cite, parent)
    for child in tree.children[3:]:
        if child.data == "reason":
            node.children.append(_build_reason(child, node))
    return node


def build_tree(parse_tree) -> list:
    """Top-level entry: parse_tree.data == 'start', children are `problem` nodes."""
    problems = []
    for prob_tree in parse_tree.children:
        node_id = str(prob_tree.children[0])
        _, confidence, _ = _extract_tags(prob_tree.children[1])  # prob_tags -> confidence only
        text = _unquote(prob_tree.children[2])
        problem_node = Node("PROBLEM", node_id, text, False, confidence, None, None)

        constraints, reason_roots, solutions, delta_text = [], [], [], None

        for block in prob_tree.children[3:]:
            inner = block.children[0]
            if inner.data == "delta":
                delta_text = _unquote(inner.children[0])
                continue
            if inner.data == "constraint":
                cid = str(inner.children[0])
                da, conf, cite = _extract_tags(inner.children[1])
                ctext = _unquote(inner.children[2])
                constraints.append(Node("CONSTRAINT", cid, ctext, da, conf, cite, problem_node))
            elif inner.data == "reason":
                reason_roots.append(_build_reason(inner, problem_node))
            elif inner.data == "solution":
                sid = str(inner.children[0])
                da, conf, cite = _extract_tags(inner.children[1])
                stext = _unquote(inner.children[2])
                solutions.append(Node("SOLUTION", sid, stext, da, conf, cite, problem_node))

        problems.append(ParsedProblem(problem_node, constraints, reason_roots, solutions, delta_text))
    return problems


# ---------------------------------------------------------------------------
# 4. Validator
# ---------------------------------------------------------------------------

class RCLError(Exception):
    pass


def _all_reason_nodes(roots):
    out = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


def compute_subtree_confidence(node) -> float:
    """Bottom-up min() propagation. Own confidence, capped by weakest descendant."""
    if not node.children:
        node.subtree_confidence = node.confidence if node.confidence is not None else 1.0
        return node.subtree_confidence
    child_min = min(compute_subtree_confidence(c) for c in node.children)
    own = node.confidence if node.confidence is not None else 1.0
    node.subtree_confidence = min(own, child_min)
    return node.subtree_confidence


def check_decide_at(pp: ParsedProblem, all_reasons):
    if not any(n.decide_at for n in all_reasons + pp.constraints + pp.solutions):
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: no [DECIDE_AT] marker found anywhere in the tree. "
            f"At least one decision point must be marked."
        )


def check_required_fields(pp: ParsedProblem, all_reasons):
    for n in pp.constraints + all_reasons + pp.solutions:
        if n.confidence is None:
            raise RCLError(f"{n.kind} {n.node_id}: missing required [confidence: ...] field.")
        if n.cite is None:
            raise RCLError(f"{n.kind} {n.node_id}: missing required [cite: ...] field.")


def check_terminal_solution(pp: ParsedProblem):
    """Every leaf REASON branch must eventually connect to a SOLUTION.
    v1 rule: at least one SOLUTION must exist and reference (via cite) a node
    that is actually present in the tree. If no SOLUTION exists at all, name
    the specific dead-end branch."""
    if not pp.solutions:
        leaves = [n for n in _all_reason_nodes(pp.reason_roots) if not n.children]
        dead_ends = ", ".join(f"{n.node_id} (\"{n.text[:40]}...\")" for n in leaves) or "no REASON nodes present"
        raise RCLError(
            f"PROBLEM {pp.problem.node_id}: unresolved — no SOLUTION found. "
            f"Dead-end leaf branch(es): {dead_ends}"
        )
    all_ids = {n.node_id for n in pp.constraints} | \
              {n.node_id for n in _all_reason_nodes(pp.reason_roots)}
    for sol in pp.solutions:
        if sol.cite not in ("UNSUPPORTED", None) and sol.cite not in all_ids:
            # cite may be free text that MENTIONS a node id, e.g. "r3, ANA test result" —
            # match on whole tokens (word boundaries) so e.g. node id "r3" doesn't get
            # falsely matched inside an unrelated word like "correct" or "barrier".
            found = any(
                re.search(rf"\b{re.escape(nid)}\b", sol.cite) for nid in all_ids
            )
            if not found:
                raise RCLError(
                    f"SOLUTION {sol.node_id}: cite \"{sol.cite}\" does not trace back to any "
                    f"node in PROBLEM {pp.problem.node_id}'s tree. A solution must be justified "
                    f"by something already established in the chain, not introduced from nowhere."
                )


def check_sibling_contradictions(pp: ParsedProblem):
    """Flag (not fail) sibling REASON nodes that reference the same constraint
    with divergent confidence in opposite directions. This is a heuristic --
    real semantic contradiction detection is out of scope for v1."""
    warnings = []
    for parent_children in [n.children for n in _all_reason_nodes(pp.reason_roots)] + [pp.reason_roots]:
        siblings = [n for n in parent_children if n.cite and n.cite != "UNSUPPORTED"]
        for i, a in enumerate(siblings):
            for b in siblings[i + 1:]:
                if a.cite == b.cite and a.confidence is not None and b.confidence is not None:
                    if abs(a.confidence - b.confidence) > 0.5:
                        warnings.append(
                            f"WARNING: {a.node_id} and {b.node_id} both cite \"{a.cite}\" "
                            f"but diverge sharply in confidence ({a.confidence} vs {b.confidence}) "
                            f"— possible contradiction, review manually."
                        )
    return warnings


def validate(pp: ParsedProblem):
    """Runs full validation. Raises RCLError with a specific message on the
    first structural failure. Returns list of non-fatal contradiction warnings
    on success."""
    all_reasons = _all_reason_nodes(pp.reason_roots)
    check_required_fields(pp, all_reasons)
    check_decide_at(pp, all_reasons)
    check_terminal_solution(pp)
    for root in pp.reason_roots:
        compute_subtree_confidence(root)
    warnings = check_sibling_contradictions(pp)
    return warnings


# ---------------------------------------------------------------------------
# 5. Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    parser = build_lark_parser("rcl_grammar.lark")
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_valid.rcl").read_text()

    tree = parser.parse(source)

    try:
        problems = build_tree(tree)
    except RCLError as e:
        print("ERROR (parse-time):", e)
        sys.exit(1)

    for pp in problems:
        print(f"\n=== PROBLEM {pp.problem.node_id}: {pp.problem.text} ===")
        try:
            warnings = validate(pp)
            print("VALID.")
            for root in pp.reason_roots:
                print(f"  {root.node_id}: declared={root.confidence} subtree_confidence={root.subtree_confidence}")
            for w in warnings:
                print(" ", w)
        except RCLError as e:
            print("ERROR:", e)
