"""Code-only proper-name variant detector. No model.

Also serves as the BASELINE ("keyword + edit distance") the LLM pipeline is compared against.
Finds capitalised names used in the scripts and groups spellings that look like variants of one
another (e.g. Millbrook / Milbrook, Marcus / Markus). A group is only a candidate: the
adjudicator (or a human) decides whether it is one entity spelled inconsistently or two real names.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz

from .parse import Episode

WORD = re.compile(r"\b[A-Z][a-z]{3,}(?:'s)?\b")
STOP = {"lady", "lord", "duke", "duchess", "dowager", "manor", "estate", "father", "mother", "sister", "brother",
        "close", "medium", "wide", "extreme", "tight", "over", "high", "low", "slow", "push", "camera", "cut",
        "scene", "episode", "continuity", "notes", "logline", "start", "end", "then", "with", "from", "this",
        "that", "your", "they", "their", "there", "what", "when", "where", "while", "please", "every", "never",
        "king", "queen", "crown", "royal", "house", "court", "church", "chapel", "cathedral", "sir", "madam",
        "master", "mistress", "god", "heaven", "christmas", "london", "england", "english", "london's"}


def name_inventory(eps: list[Episode]) -> dict[str, list[dict]]:
    """name -> occurrences [{episode, element_id, kind, quote}] for capitalised tokens that are not
    ordinary words (an ordinary word also appears in lower case somewhere in the corpus)."""
    lower_words = Counter()
    occ: dict[str, list[dict]] = defaultdict(list)
    for ep in eps:
        texts = [(None, "HEADER", ep.logline + " " + ep.continuity_notes)] + [(e.id, e.kind, e.text) for e in ep.elements]
        for eid, kind, t in texts:
            for w in re.findall(r"\b[a-z]{4,}\b", t):
                lower_words[w] += 1
            for m in WORD.finditer(t):
                w = m.group(0).removesuffix("'s")
                s = max(0, m.start() - 40)
                before = t[:m.start()].rstrip()
                initial = (not before) or before[-1] in '.!?"\u201c\u2014:'
                occ[w].append({"episode": ep.number, "element_id": eid, "kind": kind,
                               "quote": t[s:m.end() + 40].strip(), "initial": initial})
    # a capitalised word seen only at the start of sentences, at most twice, is an ordinary word
    # ('Starving and overcome...'), not a name
    return {w: o for w, o in occ.items()
            if w.lower() not in STOP and lower_words[w.lower()] == 0 and len(w) >= 4
            and not (len(o) <= 2 and all(x["initial"] for x in o))}


def variant_groups(eps: list[Episode], min_ratio: float = 75.0) -> list[dict]:
    inv = name_inventory(eps)
    names = sorted(inv, key=lambda w: (-len(inv[w]), w))
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a[0] != b[0] or a == b:
                continue
            al, bl = a.lower(), b.lower()
            if al.rstrip("s") == bl.rstrip("s"):  # plural / possessive forms
                continue
            r = fuzz.ratio(al, bl)
            shared_stem = len(al) >= 6 and len(bl) >= 6 and al[:4] == bl[:4]
            if r >= min_ratio or (shared_stem and r >= 60):
                parent[find(b)] = find(a)
    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    # second pass: two variant groups sharing a 3-letter root, each with no dominant spelling, are one drifting
    # name family (e.g. Millbrook/Milbrook + Millford/Milford); a typo group (74 vs 1) is never merged
    roots = [r for r, m in groups.items() if len(m) >= 2]
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            if find(a) == find(b) or a[:3].lower() != b[:3].lower():
                continue
            def drifting(members):  # no single dominant spelling: top count < 3x the runner-up
                c = sorted((len(inv[x]) for x in members), reverse=True)
                return c[0] < 3 * c[1]
            if drifting(groups[a]) and drifting(groups[b]):
                parent[find(b)] = find(a)
    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    out = []
    for root, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda w: -len(inv[w]))
        out.append({
            "members": [{"name": m, "count": len(inv[m]),
                         "episodes": sorted({o["episode"] for o in inv[m]}),
                         "occurrences": inv[m][:6]} for m in members],
        })
    out.sort(key=lambda g: -sum(m["count"] for m in g["members"]))
    return out
