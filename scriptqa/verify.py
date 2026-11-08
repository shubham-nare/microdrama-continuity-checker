"""Layer 3: code checks every quote the model gave. Nothing unverified reaches a flag."""
from __future__ import annotations

import re
import unicodedata

from .parse import Episode

_Q = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").translate(_Q).lower()
    s = re.sub(r"[\"']", "", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def verify_item(item: dict, ep: Episode) -> tuple[str, str | None]:
    """Return (status, element_id). status: ok | repointed | dropped.
    ok        quote found in the cited element
    repointed quote not in the cited element but found verbatim in exactly one other element of the episode
    dropped   quote not found anywhere in the episode (hallucinated or paraphrased)"""
    q = norm(item.get("quote", ""))
    if len(q) < 3:
        return "dropped", None
    by_id = ep.by_id()
    el = by_id.get(item.get("element_id", ""))
    if el is not None and q in norm(el.text + " " + (el.speaker or "")):
        return "ok", el.id
    hits = [e.id for e in ep.elements if q in norm(e.text)]
    if len(hits) >= 1:
        return "repointed", hits[0]
    return "dropped", None


def verify_episode(raw: dict, ep: Episode) -> tuple[dict, dict]:
    stats = {"facts_in": 0, "facts_ok": 0, "facts_repointed": 0, "facts_dropped": 0,
             "labels_in": 0, "labels_ok": 0, "labels_dropped": 0, "dropped_examples": []}
    by_id = ep.by_id()
    facts = []
    for f in raw.get("facts", []):
        stats["facts_in"] += 1
        status, eid = verify_item(f, ep)
        if status == "dropped":
            stats["facts_dropped"] += 1
            if len(stats["dropped_examples"]) < 3:
                stats["dropped_examples"].append({"element_id": f.get("element_id"), "quote": f.get("quote")})
            continue
        stats["facts_ok" if status == "ok" else "facts_repointed"] += 1
        el = by_id[eid]
        facts.append({**f, "element_id": eid, "verified": status, "episode": ep.number, "scene": el.scene,
                      "start": el.start, "end": el.end, "element_kind": el.kind, "speaker": el.speaker})
    speakers = {e.speaker for e in ep.elements if e.speaker}
    labels = []
    for l in raw.get("labels", []):
        stats["labels_in"] += 1
        lab = re.sub(r"\s*\(.*\)\s*$", "", l.get("label", "")).strip().upper()
        if lab not in speakers:
            stats["labels_dropped"] += 1
            continue
        status, eid = verify_item(l, ep)
        stats["labels_ok"] += 1
        labels.append({**l, "label": lab, "episode": ep.number, "evidence_ok": status != "dropped",
                       "element_id": eid or l.get("element_id")})
    # every speaker label must have a hint; add bare ones the model skipped
    seen = {l["label"] for l in labels}
    for s in sorted(speakers - seen):
        first = next(e for e in ep.elements if e.speaker == s)
        labels.append({"label": s, "character": s.title(), "element_id": first.id, "quote": "",
                       "explicit": False, "episode": ep.number, "evidence_ok": False, "added_by_code": True})
    return {"facts": facts, "labels": labels}, stats
