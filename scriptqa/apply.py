"""Close the loop for spelling flags: flag -> patch -> re-check, all in code (no model call).

For every open NAME contradiction (not dismissed by the editor), the value to keep is the editor's `canonical`
from review/decisions.yaml if given, otherwise the tool's suggestion (the spelling that needs the fewest edits).
Every whole-word occurrence of the other spellings in that flag is replaced, in copies of the scripts.
Then the pure-code spelling check runs again on the patched copies and reports what is left.

Only names are patched: a trait or injury needs a writer's judgement about which line is wrong, so those
flags stay as suggestions.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from .names import variant_groups
from .parse import parse_folder


def plan_fixes(flags: list[dict], decisions: dict) -> list[dict]:
    plan = []
    for f in flags:
        if f["type"] != "CONTRADICTION" or not (f.get("source") == "names" or f.get("kind") == "NAME"):
            continue
        d = decisions.get(f["id"], {})
        if d.get("decision") == "dismiss" or f.get("status") == "dismissed":
            continue
        g = f.get("gate") or {}
        keep = d.get("canonical") or g.get("majority_value")
        odd = [v for v in (g.get("odd_values") or []) if v != keep]
        if keep and odd:
            plan.append({"flag": f["id"], "summary": f["summary"], "keep": keep, "replace": odd,
                         "chosen_by": "editor" if d.get("canonical") else "tool suggestion (not yet accepted by an editor)"})
    return plan


def apply_fixes(data: Path, flags: list[dict], decisions: dict, out: Path) -> dict:
    plan = plan_fixes(flags, decisions)
    before = variant_groups(parse_folder(data))
    pdir = out / "patched_scripts"
    pdir.mkdir(parents=True, exist_ok=True)
    diff_lines, changed, per_flag = [], 0, {p["flag"]: 0 for p in plan}
    for src in sorted(data.glob("*.txt")):
        text = src.read_text(encoding="utf-8")
        new_lines = []
        for ln in text.splitlines(keepends=True):
            new = ln
            for p in plan:
                for o in p["replace"]:
                    new, n = re.subn(rf"\b{re.escape(o)}\b", p["keep"], new)
                    per_flag[p["flag"]] += n
            changed += new != ln
            new_lines.append(new)
        patched = "".join(new_lines)
        (pdir / src.name).write_text(patched, encoding="utf-8")
        if patched != text:
            diff_lines += difflib.unified_diff(text.splitlines(keepends=True), new_lines,
                                               f"a/{src.name}", f"b/{src.name}", n=0)
    after = variant_groups(parse_folder(pdir))

    def fmt(groups):
        return [", ".join(f"{m['name']} x{m['count']}" for m in g["members"]) for g in groups]

    rec = {"plan": plan, "lines_changed": changed, "replacements_per_flag": per_flag,
           "spelling_groups_before": fmt(before), "spelling_groups_after": fmt(after),
           "patched_scripts": str(pdir), "diff": "fix.diff"}
    (out / "fix.diff").write_text("".join(diff_lines), encoding="utf-8")
    (out / "fix_and_recheck.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec
