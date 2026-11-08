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
    a = ap.parse_args()
    result = run(a.data, a.out, a.show, a.review, a.workers)
    if a.eval:
        from eval.evaluate import evaluate_all
        ok = evaluate_all(a.data, a.out, result, a.gold)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
