"""Run the held-out fictional show with whatever model the environment selects and score it against its answer key.

    LLM_PROVIDER=openai_compat LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2:3b LLM_API_KEY=ollama \\
        python -I eval/vendor_check.py outputs/vendor_llama          # e.g. a local Meta Llama via Ollama

Uses a throwaway cache so the numbers are from a genuine live run. Writes <out>/vendor_check.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.evaluate as ev  # noqa: E402
from scriptqa.pipeline import run  # noqa: E402


def main(out: Path) -> int:
    os.environ.setdefault("LLM_CACHE_DIR", tempfile.mkdtemp(prefix="vendor_"))
    os.environ.pop("LLM_FALLBACK_MODEL", None)
    held = ROOT / "demo" / "heldout"
    ev.GOLD_DIR = held / "gold"
    t0 = time.time()
    r = run(held / "show", out, "held-out (vendor check)", None, 1, log=print)
    real = ev.score_real(r["flags"])
    al = ev.score_aliases(r["res"].get("model_assign", r["res"]["assign"]))
    s = r["summary"]
    rec = {"provider": os.environ.get("LLM_PROVIDER"), "model": os.environ.get("LLM_MODEL"),
           "planted_found": real.get("distinct_gold_found"), "planted_total": real.get("n_gold"),
           "found": {k: bool(v) for k, v in real["found"].items()}, "precision": real.get("precision"),
           "alias_f1": al["f1"], "facts_kept": s["facts_verified"], "facts_dropped_by_quote_check": s["facts_dropped"],
           "episodes_not_read": s.get("episodes_not_read"), "live_calls": r["runlog"]["usage"]["calls_live"],
           "wall_seconds": round(time.time() - t0, 1)}
    (out / "vendor_check.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "vendor_check"))
