"""End-to-end continuity pipeline. parse -> extract -> verify -> resolve -> diff/adjudicate -> report."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import diff, report
from .extract import extract_episode
from .llm import LLM
from .names import variant_groups
from .parse import parse_folder
from .resolve import resolve
from .verify import verify_episode

# Assumed paid-tier price for a Flash-Lite-class model, USD per 1M tokens (free tier costs $0).
# Only used for the "cost at volume" estimate; change in one place if pricing differs.
PRICE_IN, PRICE_OUT = 0.10, 0.40


def dump(path: Path, obj) -> None:
    # ASCII-escaped so no viewer shows mojibake for dashes and curly quotes
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=1, sort_keys=False), encoding="utf-8")


def apply_decisions(flags: list[dict], decisions: dict) -> dict:
    """review/decisions.yaml: {flag_id: {decision: dismiss|accept, canonical: '...', note: '...'}}.
    Flag ids are stable fingerprints, so an editor's decision survives re-runs."""
    stats = {"dismissed": 0, "accepted": 0, "open": 0}
    for f in flags:
        d = decisions.get(f["id"])
        if not d:
            f["status"] = "open"
        elif d.get("decision") == "dismiss":
            f["status"] = "dismissed"
        else:
            f["status"] = "accepted"
            if d.get("canonical"):
                f["suggested_fix"] = f"Editor's choice: use '{d['canonical']}'"
        if d and d.get("note"):
            f["review_note"] = d["note"]
        stats[f["status"]] += 1
    return stats


def run(data: Path, out: Path, show: str = "", review_path: Path | None = None, workers: int = 3,
        llm: LLM | None = None, log=print) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    llm = llm or LLM()
    review = yaml.safe_load(review_path.read_text(encoding="utf-8")) if review_path and review_path.exists() else {}
    eps = parse_folder(data)
    if not eps:
        raise SystemExit(f"no *.txt screenplays found in {data}")
    eps_by_num = {e.number: e for e in eps}
    show = show or data.name
    log(f"[parse] {len(eps)} episodes, {sum(len(e.elements) for e in eps)} elements")

    # 1. extract (model reads), 2. verify (code checks quotes)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        raws = list(ex.map(lambda ep: extract_episode(llm, ep), eps))
    facts, hints, vstats = [], [], {}
    for ep, raw in zip(eps, raws):
        v, st = verify_episode(raw, ep)
        facts += v["facts"]
        hints += v["labels"]
        vstats[ep.number] = st
    log(f"[extract+verify] {len(facts)} facts kept, {sum(s['facts_dropped'] for s in vstats.values())} dropped")

    # 3. resolve who is who
    res = resolve(llm, eps, hints, facts, review)
    log(f"[resolve] {len(res['characters'])} characters, {len(res['collisions'])} label collisions")

    # 4. candidates (code) -> adjudicate (model labels only) -> flags (code)
    ngroups = variant_groups(eps)
    groups = diff.build_fact_groups(facts, res) + diff.build_name_groups(ngroups, eps_by_num)
    groups = diff.adjudicate(llm, groups)
    flags = diff.dedupe_and_rank(diff.to_flags(groups, (review or {}).get("events")) + diff.label_flags(res, eps_by_num))
    dec_path = (review_path.parent / "decisions.yaml") if review_path else None
    decisions = (yaml.safe_load(dec_path.read_text(encoding="utf-8")) or {}) if dec_path and dec_path.exists() else {}
    review_stats = apply_decisions(flags, decisions.get("decisions", {}) if isinstance(decisions, dict) else {})
    report.gate_replay(eps, flags)
    log(f"[diff] {len(groups)} candidate groups -> {len(flags)} flags")

    # 5. outputs
    bible = report.build_bible(eps, facts, res, ngroups)
    cards = report.build_cards(eps, facts, res, flags)
    count = lambda t: sum(1 for f in flags if f["type"] == t)
    summary = {
        "show": show, "episodes": len(eps), "elements": sum(len(e.elements) for e in eps),
        "facts_verified": len(facts),
        "facts_dropped": sum(s["facts_dropped"] for s in vstats.values()),
        "facts_repointed": sum(s["facts_repointed"] for s in vstats.values()),
        "candidate_groups": len(groups),
        "verdicts": {k: sum(1 for g in groups if g.get("verdict", {}).get("verdict") == k)
                     for k in ("CONTRADICTION", "COMPATIBLE", "INTENTIONAL_CHANGE", "DIFFERENT_ENTITIES", "UNCLEAR")},
        "flags_total": len(flags), "flags_contradiction": count("CONTRADICTION"),
        "flags_high": sum(1 for f in flags if f["type"] == "CONTRADICTION" and f["severity"] == "high"),
        "flags_collision": count("LABEL_COLLISION"), "flags_alias": count("LABEL_ALIAS"), "flags_review": count("REVIEW"),
        "characters": len(res["characters"]),
        "review": review_stats, "human_alias_overrides": res.get("human_overrides", 0),
    }
    dump(out / "flags.json", flags)
    report.write_flags_csv(flags, out / "flags.csv")
    dump(out / "bible.json", bible)
    dump(out / "reference_cards.json", cards)
    dump(out / "aliases.json", {k: res[k] for k in ("characters", "collisions", "subject_map", "problems")})
    dump(out / "facts.json", facts)
    dump(out / "candidates.json", [{k: g[k] for k in ("group_id", "source", "kind", "subject", "attribute", "verdict")}
                                   | {"values": sorted({i["value"] for i in g["items"]})} for g in groups])
    dump(out / "verify_stats.json", vstats)
    dump(out / "summary.json", summary)
    report.write_html(out / "report.html", show, flags, bible, cards, summary)
    usage = llm.usage()
    cost = usage["in_tokens"] / 1e6 * PRICE_IN + usage["out_tokens"] / 1e6 * PRICE_OUT
    runlog = {"usage": usage, "wall_seconds": round(time.time() - t0, 1),
              "est_cost_usd_paid_tier": round(cost, 4), "price_assumption_per_M": [PRICE_IN, PRICE_OUT],
              "calls": llm.calls}
    dump(out / "run_log.json", runlog)
    if usage["calls_live"] >= 20:  # remember real (non-cached) timings for the cost section of the doc
        lr = llm.cache_dir.parent / "live_runs.json"
        hist = json.loads(lr.read_text(encoding="utf-8")) if lr.exists() else []
        hist.append({"data": str(data), "wall_seconds": runlog["wall_seconds"], "calls_live": usage["calls_live"],
                     "when": time.strftime("%Y-%m-%d")})
        lr.write_text(json.dumps(hist[-20:], indent=1), encoding="utf-8")
    log(f"[done] {summary['flags_contradiction']} contradictions ({summary['flags_high']} high), "
        f"{summary['flags_collision']} collisions, {summary['flags_alias']} alias sets; "
        f"{usage['calls_live']} live calls, {usage['in_tokens']}+{usage['out_tokens']} tokens, ~${cost:.3f} at paid prices")
    return {"summary": summary, "flags": flags, "groups": groups, "facts": facts, "res": res, "runlog": runlog}
