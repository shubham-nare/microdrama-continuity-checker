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
    a = ap.parse_args()
    if a.preflight:
        return preflight(a.data)
    if a.gate:
        return gate(a.data, a.gate, a.out, a.review)
    from scriptqa.llm import LLMError
    try:
        result = run(a.data, a.out, a.show, a.review, a.workers)
    except LLMError as e:
        print(f"\nModel call failed: {e}\n" + model_hint(str(e)))
        return 2
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
    hits = [f for f in r["flags"] if f["type"] in ("CONTRADICTION", "REVIEW", "LABEL_COLLISION")
            and any(e["episode"] == new.number for e in f["evidence"])]
    u = r["runlog"]["usage"]
    print(f"GATE episode {new.number} ({new_ep.name}) vs {len(list(data.glob('*.txt')))} earlier episodes: "
          f"{u['calls_live']} live model calls, {r['runlog']['wall_seconds']} s")
    if not hits:
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
        banner = (f"<div style='border:2px solid #c62828;padding:10px;border-radius:8px;margin:10px 0'><b>Gate result for episode {new.number}: "
                  + (f"BLOCKED by {len(hits)} new conflict(s)</b><ul>" + "".join(f"<li>{_h.escape(f['summary'])}</li>" for f in hits) + "</ul>"
                     if hits else "PASS</b>")
                  + f"<small>{len(r['flags']) - len(hits)} other flag(s) below are carried over from earlier episodes, not caused by this one.</small></div>")
        t = rep.read_text(encoding="utf-8")
        i = t.find("</h1>")
        rep.write_text(t[:i + 5] + banner + t[i + 5:], encoding="utf-8")
    import json
    gpath = out / "gate.json"
    rec = {"episode": new.number, "file": new_ep.name, "earlier_episodes": len(list(data.glob("*.txt"))),
           "live_calls": u["calls_live"], "seconds": r["runlog"]["wall_seconds"], "blocked": [f["summary"] for f in hits]}
    if u["calls_live"] == 0 and gpath.exists():  # a cached re-run: keep the last LIVE measurement for cost reporting
        old = json.loads(gpath.read_text(encoding="utf-8"))
        rec["live_calls"], rec["seconds"] = old.get("live_calls", 0), old.get("seconds", 0)
        rec["note"] = "live_calls/seconds are from the last live (uncached) gate run"
    gpath.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if hits else 0


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
        llm.cache_dir = llm.cache_dir.parent / "probe"
        llm.cache_dir.mkdir(parents=True, exist_ok=True)
        llm.complete_json("Reply with JSON.", f"Return ok=true. ({len(eps)})", Probe, tag="preflight")
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
