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


def pct(x):
    return "n/a" if x is None else f"{round(100 * x)}%"


def live_minutes(run: Path) -> str:
    """Wall time of the most recent LIVE run (cached re-runs take seconds), from .cache/live_runs.json."""
    p = ROOT / ".cache" / "live_runs.json"
    if p.exists():
        runs = json.loads(p.read_text(encoding="utf-8"))
        if runs:
            r = runs[-1]
            return f"{r['wall_seconds'] / 60:.0f} minutes ({r['calls_live']} live calls, measured {r['when']})"
    return "not measured"


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
    found = [titles.get(k, k) + (f" ({cov[k]['found']}/{cov[k]['of']} variants)" if k in cov and cov[k]["of"] > 2 else "")
             for k, v in real["found"].items() if v]
    missed = [titles.get(k, k) for k, v in real["found"].items() if not v]
    n_coll = sum(1 for k in real["found"] if "collision" in k)
    flags = json.loads((run / "flags.json").read_text(encoding="utf-8"))
    gates = [f for f in flags if f.get("gate", {}).get("blocked_at_episode")]
    top_gate = max(gates, key=lambda f: f["gate"]["later_dialogue_lines"] + f["gate"]["later_camera_shots_in_affected_scenes"], default=None)
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
        "held_found": (e.get("heldout") or {}).get("real", {}).get("distinct_gold_found", "n/a"),
        "held_n": (e.get("heldout") or {}).get("real", {}).get("n_gold", "n/a"),
        "held_ci": "–".join(f"{round(100 * x)}%" for x in (e.get("heldout") or {}).get("ci95", [0, 0])),
        "held_prec": pct((e.get("heldout") or {}).get("real", {}).get("precision")),
        "held_missed": ", ".join(k.replace("_", " ") for k, v in (e.get("heldout") or {}).get("real", {}).get("found", {}).items() if not v) or "none",
        "loss_text": "; ".join(f"{titles.get(k, k)}: {v}" for k, v in real.get("loss_stage", {}).items() if v != "flagged") or "none",
        "gate_text": (f"For example, the spelling family {', '.join(top_gate['gate']['odd_values'][:4])}… would have been blocked at episode "
                      f"{top_gate['gate']['blocked_at_episode']}. Missing it lets the odd spellings into {top_gate['gate']['later_dialogue_lines']} later spoken lines "
                      f"and {top_gate['gate']['later_camera_shots_in_affected_scenes']} camera shots in those scenes." if top_gate else ""),
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
