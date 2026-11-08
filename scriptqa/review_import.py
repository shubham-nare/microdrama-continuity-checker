"""Turn an editor's filled-in flags.csv into review/decisions.yaml, which the next run applies.

    python -m scriptqa.review_import outputs/run1/flags.csv            # merges into review/decisions.yaml

Columns read: id, "reviewer_decision (dismiss/accept)", "canonical (if accept)". Flag ids are stable
fingerprints, so decisions survive re-runs and only new flags show up as open.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def import_csv(csv_path: Path, out: Path = ROOT / "review" / "decisions.yaml") -> dict:
    data = (yaml.safe_load(out.read_text(encoding="utf-8")) or {}) if out.exists() else {}
    decisions = data.get("decisions", {}) or {}
    n = 0
    with csv_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            d = (row.get("reviewer_decision (dismiss/accept)") or "").strip().lower()
            if d not in ("dismiss", "accept"):
                continue
            entry = {"decision": d}
            canon = (row.get("canonical (if accept)") or "").strip()
            if canon:
                entry["canonical"] = canon
            decisions[row["id"]] = entry
            n += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"decisions": decisions}, allow_unicode=True, sort_keys=True), encoding="utf-8")
    print(f"imported {n} decisions -> {out} ({len(decisions)} total)")
    return decisions


if __name__ == "__main__":
    import_csv(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "review" / "decisions.yaml")
