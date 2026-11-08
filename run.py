"""Single entry command.

    python run.py --data data --out outputs/iter_01            # continuity check on a show folder
    python run.py --data path/to/new_show --out outputs/new     # any new show in the same format
    python run.py ... --eval                                     # + seeded-error eval, gold scoring, deterministic checks

Model: set LLM_PROVIDER / LLM_MODEL (and keys) in .env or the environment. Nothing else changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scriptqa.pipeline import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuity checker for AI-written microdrama scripts")
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="folder of episode .txt screenplays")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "latest")
    ap.add_argument("--show", default="", help="show name for the report")
    ap.add_argument("--review", type=Path, default=ROOT / "review" / "aliases.yaml",
                    help="human-confirmed alias/intentional-change decisions (optional)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval", action="store_true", help="also run seeded-error eval, gold scoring and checks")
    ap.add_argument("--gold", type=Path, default=ROOT / "eval", help="folder with gold_real.yaml, gold_aliases.yaml, flag_labels.yaml")
    ap.add_argument("--preflight", action="store_true", help="check the input and the model connection, then stop")
    ap.add_argument("--gate", type=Path, default=None, help="check ONE new episode file against the show in --data")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved cache: a real live run into a throwaway cache")
    ap.add_argument("--code-only", action="store_true", help="no model at all: parse + spelling-variant check + speaker labels, in seconds")
    ap.add_argument("--apply", type=Path, default=None, metavar="FLAGS_JSON",
                    help="patch the scripts for the spelling flags in FLAGS_JSON (editor's choice or the suggestion), then re-check; no model")
    a = ap.parse_args()
    if a.fresh:
        import os
        import tempfile
        os.environ["LLM_CACHE_DIR"] = tempfile.mkdtemp(prefix="scriptqa_fresh_")
        print(f"[fresh] live run, throwaway cache at {os.environ['LLM_CACHE_DIR']}")
    if a.preflight:
        return preflight(a.data)
    if a.gate:
        return gate(a.data, a.gate, a.out, a.review)
    if a.code_only:
        return code_only(a.data, a.out)
    if a.apply:
        import json
        return fix_and_recheck(a.data, json.loads(a.apply.read_text(encoding="utf-8")), a.out, a.review)
    from scriptqa.llm import LLMError
    try:
        result = run(a.data, a.out, a.show, a.review, a.workers)
    except LLMError as e:
        print(f"\nModel call failed: {e}\n" + model_hint(str(e)))
        return 2
    s = result["summary"]
    if s.get("episodes_not_read") and len(s["episodes_not_read"]) == s["episodes"]:
        print("\nNo episode was read by the model. " + model_hint(str(s.get("extract_errors", [{}])[0].get("error", ""))))
        return 2
    fix_and_recheck(a.data, result["flags"], a.out / "fix", a.review)
    if a.eval:
        from eval.evaluate import evaluate_all
        ok = evaluate_all(a.data, a.out, result, a.gold)
        return 0 if ok else 1
    return 0


def gate(data: Path, new_ep: Path, out: Path, review: Path) -> int:
    """Per-episode gate: the show so far + one new episode. Old episodes come from the cache, so a gate costs
    about 1 extraction call plus re-resolution and re-adjudication. Prints BLOCKED for every flag that touches
    the new episode, with the bible's value and the new line."""
    import shutil
    import tempfile
    from scriptqa.parse import parse_file
    new = parse_file(new_ep)
    work = Path(tempfile.mkdtemp(prefix="gate_"))
    for p in data.glob("*.txt"):
        if p.resolve() != new_ep.resolve():
            shutil.copy(p, work / p.name)
    shutil.copy(new_ep, work / new_ep.name)
    r = run(work, out, f"gate: episode {new.number}", review, 3, log=lambda *a: None)
    from collections import Counter
    touching = [f for f in r["flags"] if f["type"] in ("CONTRADICTION", "REVIEW", "LABEL_COLLISION")
                and any(e["episode"] == new.number for e in f["evidence"])]
    hits, ties, agrees = [], [], []
    for f in touching:
        if f["type"] == "LABEL_COLLISION":
            hits.append(f)
            continue
        def sig(v):  # compare the words that make two descriptions disagree (colour, material, side), else the value
            import re as _re
            from scriptqa.diff import CLASH_WORDS
            ws = _re.split(r"[\s,;/-]+", v.lower())
            key = {CLASH_WORDS[w] for w in ws if w in CLASH_WORDS} | {w for w in ws if w in ("left", "right")}
            return " ".join(sorted(key)) or v.lower()
        earlier = Counter(sig(e["value"]) for e in f["evidence"] if e["episode"] != new.number)
        mine = {sig(e["value"]) for e in f["evidence"] if e["episode"] == new.number}
        top = earlier.most_common()
        if not top:
            hits.append(f)
            continue
        best = top[0][1]
        leaders = {v for v, n in top if n == best}
        if mine - set(earlier):
            hits.append(f)            # the new episode introduces a value never used before
        elif mine <= leaders and len(leaders) == 1:
            agrees.append(f)          # the new episode follows the established value; the odd line is older
        else:
            ties.append(f)            # earlier episodes already disagree; the new episode sides with one of them
    u = r["runlog"]["usage"]
    print(f"GATE episode {new.number} ({new_ep.name}) vs {len(list(data.glob('*.txt')))} earlier episodes: "
          f"{u['calls_live']} live model calls, {r['runlog']['wall_seconds']} s")
    for f in ties:
        print(f"WARN (earlier episodes already disagree; this one sides with one of them) {f['summary']}")
    if agrees:
        print(f"note: {len(agrees)} older conflict(s) touch this episode, but it follows the established value")
    if not hits and ties:
        print(f"NEEDS SIGN-OFF: this episode repeats {len(ties)} value(s) that earlier episodes already disagree on; an editor must pick one")
    elif not hits:
        print("PASS: no continuity conflict involving this episode")
    for f in hits:
        mine = [e for e in f["evidence"] if e["episode"] == new.number]
        before = [e for e in f["evidence"] if e["episode"] != new.number]
        print(f"BLOCKED [{f['severity']}] {f['summary']}")
        for e in mine[:2]:
            print(f"   new  ep{e['episode']} line {e.get('line')}: {e['value']!r} \"{e['quote'][:90]}\"")
        for e in before[:2]:
            print(f"   show ep{e['episode']} line {e.get('line')}: {e['value']!r} \"{e['quote'][:90]}\"")
    import html as _h
    rep = out / "report.html"
    if rep.exists():
        colour = "#c62828" if hits else ("#ef6c00" if ties else "#2e7d32")
        verdict = "BLOCKED" if hits else ("NEEDS SIGN-OFF" if ties else "PASS")
        carried = sum(1 for f in r["flags"] if f["type"] in ("CONTRADICTION", "REVIEW", "LABEL_COLLISION")) - len(hits)
        banner = (f"<div style='border:2px solid {colour};padding:10px;border-radius:8px;margin:10px 0'><b>Gate result for episode {new.number}: "
                  + (f"{len(ties)} warning(s) where earlier episodes already disagree. " if ties else "")
                  + "Blocked = this episode introduces a value the show never used. "
                  + (f"BLOCKED by {len(hits)} new conflict(s)</b><ul>" + "".join(f"<li>{_h.escape(f['summary'])}</li>" for f in hits) + "</ul>"
                     if hits else (f"NEEDS SIGN-OFF</b> (repeats a disputed value)<ul>" + "".join(f"<li>{_h.escape(f['summary'])}</li>" for f in ties) + "</ul>"
                                   if ties else "PASS</b>"))
                  + f"<small>{carried} other flag(s) below are carried over from earlier episodes, not caused by this one.</small></div>")
        t = rep.read_text(encoding="utf-8")
        i = t.find("</h1>")
        rep.write_text(t[:i + 5] + banner + t[i + 5:], encoding="utf-8")
    import json
    gpath = out / "gate.json"
    rec = {"episode": new.number, "file": new_ep.name, "earlier_episodes": len(list(data.glob("*.txt"))),
           "live_calls": u["calls_live"], "seconds": r["runlog"]["wall_seconds"], "blocked": [f["summary"] for f in hits],
           "verdict": "BLOCKED" if hits else ("NEEDS SIGN-OFF" if ties else "PASS"),
           "warn_earlier_disagree": [f["summary"] for f in ties], "older_conflicts_followed": [f["summary"] for f in agrees],
           "evidence": [{"flag": f["id"], "new": [{"line": e["line"], "value": e["value"], "quote": e["quote"]} for e in f["evidence"] if e["episode"] == new.number],
                         "earlier": [{"episode": e["episode"], "line": e["line"], "value": e["value"], "quote": e["quote"]} for e in f["evidence"] if e["episode"] != new.number][:3]}
                        for f in hits]}
    if u["calls_live"] == 0 and gpath.exists():  # a cached re-run: keep the last LIVE measurement for cost reporting
        old = json.loads(gpath.read_text(encoding="utf-8"))
        rec["live_calls"], rec["seconds"] = old.get("live_calls", 0), old.get("seconds", 0)
        rec["note"] = "live_calls/seconds are from the last live (uncached) gate run"
    gpath.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if (hits or ties) else 0


def fix_and_recheck(data: Path, flags: list, out: Path, review: Path) -> int:
    """Flag -> patch -> re-check for spelling flags, in code. Writes fix.diff, patched_scripts/ and fix_and_recheck.json."""
    import yaml
    from scriptqa.apply import apply_fixes
    dec_path = review.parent / "decisions.yaml" if review else None
    dec = (yaml.safe_load(dec_path.read_text(encoding="utf-8")) or {}) if dec_path and dec_path.exists() else {}
    rec = apply_fixes(data, flags, (dec or {}).get("decisions") or {}, out)
    rep = out.parent / "report.html"
    if rep.exists():  # add the result to the editor's report (generated, not hand-written)
        import html as _h
        applied = [p for p in rec["plan"] if p["status"] == "applied"]
        held = [p for p in rec["plan"] if p["status"].startswith("HELD")]
        pv = rec.get("preview_one_spelling")
        sec = ("<h2>Fix and re-check (spelling flags)</h2><p>Applied (flags an editor accepted): "
               + (", ".join(f"{_h.escape(', '.join(p['replace']))} &rarr; <b>{_h.escape(p['keep'])}</b>" for p in applied) or "nothing")
               + f" ({rec['lines_changed']} script lines, dialogue and directions: fix/fix.diff).</p>"
               + ("<p><b>Held</b> (the two suggestions are different spellings of what may be one name; patching both would "
                  "write a new contradiction): " + "; ".join(f"{_h.escape(p['flag'])} would keep '{_h.escape(p['keep'])}'" for p in held)
                  + ". Choose one spelling and set it as <i>canonical</i> for both in review/decisions.yaml.</p>" if held else "")
               + (f"<p>Preview, not applied (every suggestion accepted"
                  + (f", '{_h.escape(pv['shared_value'])}' for the clashing flags" if pv['shared_value'] != "n/a" else "")
                  + f"): {pv['lines_changed']} lines change and "
                  f"the spelling check then finds {len(pv['spelling_groups_after'])} group(s) left"
                  + (": " + "; ".join(_h.escape(g) for g in pv["spelling_groups_after"]) if pv["spelling_groups_after"] else "")
                  + " (fix/preview_one_spelling/fix.diff).</p>" if pv else "")
               + "<p><small>Spelling check on the patched copy now: "
               + ("; ".join(_h.escape(g) for g in rec["spelling_groups_after"]) or "nothing left")
               + ". Dismissed flags are never patched; traits and injuries are not patched (which line is wrong is a writer's call)."
               "</small></p>")
        t = rep.read_text(encoding="utf-8")
        anchor = "<h2>Cleared by the model" if "<h2>Cleared by the model" in t else "</body>"
        rep.write_text(t.replace(anchor, sec + anchor, 1), encoding="utf-8")
    n_app = sum(p["status"] == "applied" for p in rec["plan"])
    print(f"[fix] {n_app} accepted spelling flag(s) patched, {len(rec['plan']) - n_app} waiting for an editor, {rec['lines_changed']} script lines changed -> {out / 'fix.diff'}")
    for p in rec["plan"]:
        if p["status"].startswith("HELD"):
            print(f"      {p['status']}")
    if rec.get("preview_one_spelling"):
        pv = rec["preview_one_spelling"]
        print(f"[fix] preview (not applied; all suggestions, '{pv['shared_value']}' for clashes): {pv['lines_changed']} lines, "
              f"{len(pv['spelling_groups_after'])} spelling group(s) left")
    print(f"[fix] spelling groups before: {len(rec['spelling_groups_before'])}, after the patch: {len(rec['spelling_groups_after'])}"
          + "".join(f"\n      still spelled several ways: {g}" for g in rec["spelling_groups_after"]))
    return 0


def code_only(data: Path, out: Path) -> int:
    """Interview safety net: what code alone finds (spelling variants, speaker labels) with zero model calls."""
    import json
    from scriptqa.names import variant_groups
    from scriptqa.parse import parse_folder
    eps = parse_folder(data)
    groups = variant_groups(eps)
    labels = {}
    for e in eps:
        for el in e.elements:
            if el.speaker:
                labels.setdefault(el.speaker, set()).add(e.number)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"episodes": len(eps), "elements": sum(len(e.elements) for e in eps),
           "spelling_groups": [[{"name": m["name"], "count": m["count"], "episodes": m["episodes"]} for m in g["members"]] for g in groups],
           "speaker_labels": {k: sorted(v) for k, v in sorted(labels.items())}}
    (out / "code_only.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[code-only] {rec['episodes']} episodes, {rec['elements']} lines, {len(labels)} speaker labels, no model calls")
    for g in rec["spelling_groups"]:
        print("  spelled several ways: " + ", ".join(f"{m['name']} x{m['count']}" for m in g))
    print(f"  wrote {out / 'code_only.json'} (the model layer was skipped: no trait, prop, injury or who-is-who checks)")
    return 0


def model_hint(msg: str) -> str:
    m = msg.lower()
    if "not set" in m or "api key" in m or "401" in m or "403" in m or "permission" in m:
        return "Fix: put the key for LLM_PROVIDER in .env (GEMINI_API_KEY, or LLM_API_KEY + LLM_BASE_URL for openai_compat)."
    if "404" in m or "not found" in m:
        return "Fix: the model name is not available to this key. Check LLM_MODEL (and LLM_BASE_URL for openai_compat)."
    if "503" in m or "unavailable" in m or "overloaded" in m or "429" in m or "resource_exhausted" in m:
        return "The provider is busy or the quota is used up. Retry later, or set LLM_FALLBACK_MODEL / another LLM_MODEL. Cached episodes are kept."
    return "Run `python run.py --preflight --data <folder>` to check the setup."


def preflight(data: Path) -> int:
    """No-surprise check before a live run (e.g. a new show in an interview): input format, model, key, one probe call."""
    import json
    from pydantic import BaseModel
    from scriptqa.llm import LLM, LLMError
    from scriptqa.parse import parse_folder
    eps = parse_folder(data)
    if not eps:
        print(f"FAIL no *.txt screenplays in {data}")
        return 2
    bad = [e.number for e in eps if len(e.elements) != e.raw_timestamp_lines]
    labels = sorted({el.speaker for e in eps for el in e.elements if el.speaker})
    print(f"input  : {len(eps)} episodes, {sum(len(e.elements) for e in eps)} timed elements, {len(labels)} speaker labels")
    print(f"format : {'OK' if not bad else 'WARNING element/timestamp mismatch in episodes ' + str(bad)}")
    llm = LLM()
    print(f"model  : provider={llm.provider} model={llm.model} fallback={llm.fallback}")

    class Probe(BaseModel):
        ok: bool

    try:
        import tempfile
        import time as _t
        llm.cache_dir = Path(tempfile.mkdtemp(prefix="probe_"))  # never cached: a real call every time
        llm.complete_json("Reply with JSON.", f"Return ok=true. ({_t.time()})", Probe, tag="preflight")
        print("probe  : OK (one live call)")
    except LLMError as e:
        print(f"probe  : FAIL {e}\n         {model_hint(str(e))}")
        return 2
    lr = ROOT / ".cache" / "live_runs.json"
    per_call = 6.6
    if lr.exists():
        runs = json.loads(lr.read_text(encoding="utf-8"))
        if runs:
            per_call = runs[-1]["wall_seconds"] / max(1, runs[-1]["calls_live"])
    calls = len(eps) + 1 + max(1, len(eps) // 8)
    print(f"plan   : about {calls} model calls, about {calls * per_call / 60:.0f} min live (cached episodes are free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
