"""One genuine live run on a show folder with an empty cache, scored against the answer key in --gold.

    python -I eval/fresh_full.py data outputs/fresh_full          # writes fresh_score.json
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
os.environ["LLM_CACHE_DIR"] = tempfile.mkdtemp(prefix="fresh_full_")

import eval.evaluate as ev  # noqa: E402
from scriptqa.pipeline import run  # noqa: E402


def main(data: Path, out: Path) -> int:
    t0 = time.time()
    r = run(data, out, "fresh full run", ROOT / "review" / "aliases.yaml", 3, log=lambda *a: print(*a, flush=True))
    real = ev.score_real(r["flags"])
    al = ev.score_aliases(r["res"].get("model_assign", r["res"]["assign"]))
    rec = {"real_found": real.get("distinct_gold_found"), "real_n": real.get("n_gold"),
           "found": {k: bool(v) for k, v in real["found"].items()}, "precision": real.get("precision"),
           "n_scored": real.get("n_flags_scored"), "unlabelled_ids": [j["id"] for j in real["judged"] if j["label"] == "unlabelled"],
           "characters_fully_merged": al.get("characters_fully_merged"), "live_calls": r["runlog"]["usage"]["calls_live"],
           "wall_seconds": round(time.time() - t0, 1), "cache_dir": os.environ["LLM_CACHE_DIR"]}
    (out / "fresh_score.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
