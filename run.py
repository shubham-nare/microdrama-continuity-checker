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
    a = ap.parse_args()
    if a.preflight:
        return preflight(a.data)
    from scriptqa.llm import LLMError
    try:
        result = run(a.data, a.out, a.show, a.review, a.workers)
    except LLMError as e:
        print(f"\nModel call failed: {e}\nFix: put the key for LLM_PROVIDER in .env (GEMINI_API_KEY, or LLM_API_KEY + LLM_BASE_URL "
              "for openai_compat), then run `python run.py --preflight --data <folder>`.")
        return 2
    if a.eval:
        from eval.evaluate import evaluate_all
        ok = evaluate_all(a.data, a.out, result, a.gold)
        return 0 if ok else 1
    return 0


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
        print(f"probe  : FAIL {e}")
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
