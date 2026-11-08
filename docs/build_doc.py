"""Build the 2-page solution doc (HTML -> PDF via headless Chrome/Edge) with every number read from a run folder.

    python docs/build_doc.py outputs/iter_02
The prose lives in docs/solution_template.html; {placeholders} are filled from summary.json, eval.json and
run_log.json, so the doc can never quote a number the outputs don't contain.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


PLAIN = {4: "who-is-who (only 5 of 9 multi-label characters fully merged; bar 80%)",
         5: "planted errors caught (bar 70%, with at most 2 new false flags)",
         6: "share of flags that are real errors (bar 70%)",
         11: "real errors found (bar 70%)"}


def pct(x):
    return "n/a" if x is None else f"{round(100 * x)}%"


def live_minutes(run: Path) -> str:
    """Wall time of the most recent LIVE run (cached re-runs take seconds), from .cache/live_runs.json."""
    p = ROOT / ".cache" / "live_runs.json"
    if p.exists():
        runs = json.loads(p.read_text(encoding="utf-8"))
        runs = [r for r in runs if Path(r["data"]).name == "data" and "mutated" not in r["data"]]
        if runs:
            r = max(runs, key=lambda r: r["calls_live"])  # the fullest live run (nothing cached) is the honest timing
            return f"{r['wall_seconds'] / 60:.0f} minutes ({r['calls_live']} live calls, measured {r['when']})"
    return "not measured"


def extra_vals(real, seeded, flags) -> dict:
    by_id = {f["id"]: f for f in flags}
    judged = [j for j in real["judged"] if j.get("type") != "LABEL_ALIAS"]
    false_ex = [by_id[j["id"]]["summary"].split(":")[0].replace("—", "-").replace("* -", "unattributed").replace("Unattributed -", "unattributed")
                for j in judged if j["label"] == "false" and j["id"] in by_id]

    def prec(types):
        js = [j for j in judged if j.get("type") in types and j["label"] != "unlabelled"]
        return f"{sum(1 for j in js if j['label'] == 'true')}/{len(js)}" if js else "n/a"
    stab = []
    for i in (1, 2, 3):
        p = ROOT / "outputs" / "stability" / f"run{i}" / "vendor_check.json"
        if p.exists():
            stab.append(json.loads(p.read_text(encoding="utf-8")))
    common = "n/a"
    if len(stab) == 3:
        sets = [set(f["id"] for f in json.loads((ROOT / "outputs" / "stability" / f"run{i}" / "flags.json").read_text(encoding="utf-8"))
                    if f["type"] != "LABEL_ALIAS") for i in (1, 2, 3)]
        common = str(len(sets[0] & sets[1] & sets[2]))
    missed = set.intersection(*[{k for k, v in s["found"].items() if not v} for s in stab]) if stab else set()
    lp = ROOT / "outputs" / "vendor_llama" / "vendor_check.json"
    ll = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    loss = {k: v for k, v in real.get("loss_stage", {}).items() if v != "flagged"}
    rel = seeded.get("per_type", {}).get("relation_flip", {})
    return {
        "false_examples": "; ".join(false_ex) or "none",
        "prec_contra": prec(("CONTRADICTION", "LABEL_COLLISION")), "prec_review": prec(("REVIEW",)),
        "stab_found": ", ".join(f"{s['planted_found']}/{s['planted_total']}" for s in stab) or "not run",
        "stab_common": common, "stab_missed": ", ".join(sorted(m.replace("_", " ") for m in missed)) or "none",
        "llama_found": ll.get("planted_found", "n/a"), "llama_dropped": ll.get("facts_dropped_by_quote_check", "n/a"),
        "llama_facts": (ll.get("facts_kept", 0) + ll.get("facts_dropped_by_quote_check", 0)) if ll else "n/a",
        "loss_text_short": "; ".join(f"{k.replace('_', ' ')} ({v.split(':')[0].split(' as ')[0]})" for k, v in loss.items()) or "none",
        "relation_rate": f"{rel.get('pipeline', 0)}/{rel.get('n', 0)} caught" if rel else "n/a",
    }


def main(run: Path) -> int:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    e = json.loads((run / "eval.json").read_text(encoding="utf-8"))
    rl = json.loads((run / "run_log.json").read_text(encoding="utf-8"))
    checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
    demo = json.loads((ROOT / "demo" / "output" / "summary.json").read_text(encoding="utf-8")) if (ROOT / "demo" / "output" / "summary.json").exists() else {}
    seeded, real, al = e["seeded"], e["real"], e["aliases"]
    pt = seeded["per_type"]
    per_type = ", ".join(f"{k.replace('_', ' ')} {v['pipeline']}/{v['n']} (baseline {v['baseline']})" for k, v in sorted(pt.items()))
    import yaml
    gold_path = ROOT / "eval" / "gold_real.yaml"
    titles = {g["id"]: g.get("title", g["id"].replace("_", " "))
              for g in (yaml.safe_load(gold_path.read_text(encoding="utf-8"))["items"] if gold_path.exists() else [])}
    cov = real.get("coverage", {})
    found = [titles.get(k, k) + (f" ({cov[k]['found']}/{cov[k]['of']} variants)" if k in cov and cov[k]["of"] > 2 and "(" not in titles.get(k, k) else "")
             for k, v in real["found"].items() if v]
    missed = [titles.get(k, k) for k, v in real["found"].items() if not v]
    n_coll = sum(1 for k in real["found"] if "collision" in k)
    flags = json.loads((run / "flags.json").read_text(encoding="utf-8"))
    true_ids = {j["id"] for j in real["judged"] if j["label"] == "true"}
    gates = [f for f in flags if f.get("gate", {}).get("blocked_at_episode") and f["id"] in true_ids and f["gate"].get("odd_values")]
    gates = [f for f in gates if f["gate"]["later_dialogue_lines"] > 0]
    top_gate = max(gates, key=lambda f: f["gate"]["later_dialogue_lines"], default=None)
    swap = next((c["detail"] for c in checks if c["n"] == 8), "")
    canary = swap.split("canary: ")[1] if "canary: " in swap else "not run"
    false_flags = [j for j in real["judged"] if j["label"] == "false" and j.get("type") != "LABEL_ALIAS"]
    u = rl["usage"]
    tokens = u["in_tokens"] + u["out_tokens"]
    cost = rl["est_cost_usd_paid_tier"]
    vals = {
        "run": run.name, "episodes": s["episodes"], "elements": s["elements"], "facts": s["facts_verified"],
        "dropped": s["facts_dropped"], "flags_total": s["flags_total"], "contradictions": s["flags_contradiction"],
        "high": s["flags_high"], "collisions": s["flags_collision"], "aliases": s["flags_alias"],
        "real_recall": pct(real["recall"]), "real_found": len(found), "real_n": len(real["found"]),
        "real_found_list": ", ".join(found), "real_missed_list": ", ".join(missed) or "none",
        "real_ci": "–".join(f"{round(100 * x)}%" for x in real.get("ci95", [0, 0])),
        "seed_ci": "–".join(f"{round(100 * x)}%" for x in seeded.get("ci95", [0, 0])),
        "seed_distinct": seeded.get("distinct_detecting_flags", "n/a"), "seed_caught": seeded.get("caught", "n/a"),
        "canary": canary,
        **extra_vals(real, seeded, flags),
        "fresh_text": (lambda d: f" A fresh full live run (empty cache, {d['live_calls']} calls, {d['wall_seconds'] / 60:.0f} min) found {d['real_found']}/{d['real_n']}."
                       if d else "")(json.loads((ROOT / "outputs" / "fresh_full" / "fresh_score.json").read_text(encoding="utf-8"))
                                     if (ROOT / "outputs" / "fresh_full" / "fresh_score.json").exists() else None),
        "canary_plain": canary.replace("swapped model catches", "caught").replace(" in the demo show", "").split(" (")[0] if canary != "not run" else "was not run",
        "checks_failed": "; ".join(PLAIN.get(c["n"], c["check"].split(":")[0]) for c in checks if not c["pass"]) or "none",
        "seed_valid": ", ".join(f"{v['caught']}/{v['n']} {k}" for k, v in sorted(seeded.get("by_validity", {}).items())),
        "gate_demo": (lambda g: (f"On a never-seen episode of the held-out show it took {g['live_calls']} live model call{'s' if g['live_calls'] != 1 else ''}"
                                 + (f" ({g['seconds'] / 60:.0f} min, while the free tier was rate-limiting)" if g['seconds'] > 120 else f" and {g['seconds']:.0f} s")
                                 + (f" and blocked {len(g['blocked'])} conflict{'s' if len(g['blocked']) != 1 else ''} the new episode touches." if g.get("blocked") else " and passed."))
                      if g else "")(json.loads((ROOT / "demo" / "gate_output_new" / "gate.json").read_text(encoding="utf-8")) if (ROOT / "demo" / "gate_output_new" / "gate.json").exists() else None),
        "held_found": (e.get("heldout") or {}).get("real", {}).get("distinct_gold_found", "n/a"),
        "held_n": (e.get("heldout") or {}).get("real", {}).get("n_gold", "n/a"),
        "held_ci": "–".join(f"{round(100 * x)}%" for x in (e.get("heldout") or {}).get("ci95", [0, 0])),
        "held_prec": pct((e.get("heldout") or {}).get("real", {}).get("precision")),
        "held_missed": ", ".join(k.replace("_", " ") for k, v in (e.get("heldout") or {}).get("real", {}).get("found", {}).items() if not v) or "none",
        "loss_text": "; ".join(f"{titles.get(k, k)}: {v.split(':')[0].replace('INTENTIONAL_CHANGE', 'an intentional change').replace('DIFFERENT_ENTITIES', 'different things').replace('COMPATIBLE', 'compatible')}"
                               for k, v in real.get("loss_stage", {}).items() if v != "flagged") or "none",
        "alias_chars": f"{al.get('characters_fully_merged', '?')}/{al.get('characters_with_several_labels', '?')}",
        "alias_misses": ", ".join(f"{c} ({v})" for c, v in al.get("per_character", {}).items() if v != "fully merged") or "none",
        "gate_text": (f"On Kuku's pack, checking each episode as written would have stopped “{top_gate['summary'].split(':')[0]}” at episode "
                      f"{top_gate['gate']['blocked_at_episode']}, before {top_gate['gate']['later_dialogue_lines']} later spoken lines used a minority form "
                      "(lines to re-record)." if top_gate else ""),
        "precision": pct(real["precision"]), "n_scored": real["n_flags_scored"], "n_false": f"{len(false_flags)} {'is' if len(false_flags) == 1 else 'are'}",
        "seed_recall": pct(seeded["pipeline_recall"]), "base_recall": pct(seeded["baseline_recall"]),
        "seed_n": seeded["n"], "per_type": per_type, "spurious": (lambda n: f"{n} spurious new flag" + ("" if n == 1 else "s"))(len(seeded["spurious_new_flags"])),
        "alias_f1": al["f1"], "alias_p": al["pairwise_precision"], "alias_r": al["pairwise_recall"],
        "calls": u["calls_total"], "tokens_k": round(tokens / 1000), "cost": f"{cost:.3f}",
        "cost_month_150": f"{cost * 150:.0f}", "cost_month_1000": f"{cost * 1000:.0f}",
        "checks_passed": sum(c["pass"] for c in checks), "checks_n": len(checks),
        "demo_flags": demo.get("flags_total", "n/a"),
        "live_min": live_minutes(run),
        "alias_flag_precision": (f"{sum(1 for j in real['judged'] if j.get('type') == 'LABEL_ALIAS' and j['label'] == 'true')}/{real.get('n_alias_flags', 0)}"), "n_alias": "",
        "collision_text": (f"The {'one' if n_coll == 1 else n_coll} hand-labelled label collision{'s are' if n_coll > 1 else ' is'} found by the model and code alone, before any human input."
                           if all(real["found"].get(k) for k in real["found"] if "collision" in k)
                           else "The hand-labelled label collision is missed: one label for two people is merged into one."),
    }
    tpl = (ROOT / "docs" / "solution_template.html").read_text(encoding="utf-8")
    html = tpl.format(**vals)
    import re as _re
    body = _re.sub(r"<style.*?</style>", " ", html, flags=_re.S)
    words = len(_re.sub(r"<[^>]+>", " ", body).split())
    gold_ids = [g for g in titles if "_" in g]
    problems = []
    if words > 1000:
        problems.append(f"{words} words (budget 1000)")
    if any(gid in body for gid in gold_ids):
        problems.append("an internal answer-key id leaked into the doc")
    if _re.search(r"ep\d+->ep\d+", body):
        problems.append("a raw 'epNN->epNN' string leaked into the doc")
    if _re.search(r"font-size:\s*[0-9](\.\d+)?pt", tpl):
        problems.append("a font below 10 pt")
    print(f"doc budget: {words} words" + ("; PROBLEMS: " + "; ".join(problems) if problems else "; ok"))
    if problems:
        return 3
    out_html = ROOT / "docs" / "solution.html"
    out_html.write_text(html, encoding="utf-8")
    pdf = ROOT / "docs" / "solution.pdf"
    browser = next((b for b in [shutil.which("chrome"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                shutil.which("chromium"), shutil.which("google-chrome")] if b and Path(b).exists()), None)
    if not browser:
        print("no Chrome/Edge found; open docs/solution.html and print to PDF")
        return 1
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True, timeout=120)
    from pypdf import PdfReader
    pages = len(PdfReader(str(pdf)).pages)
    print(f"wrote {pdf} ({pages} pages) from {run}")
    return 0 if pages <= 2 else 2


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "latest"))
