"""Close the loop for spelling flags: flag -> patch -> re-check, all in code (no model call).

Only flags an editor ACCEPTED in review/decisions.yaml are patched (human confirms); the value kept is the editor's
`canonical` if given, otherwise the tool's suggestion (the spelling that needs the fewest edits). Every other open
spelling flag goes to a preview (preview_one_spelling/), which is written but never applied.
Every whole-word occurrence of the other spellings in that flag is replaced, in copies of the scripts.
Then the pure-code spelling check runs again on the patched copies and reports what is left.

Safety rule: when two flags would keep DIFFERENT spellings of what may be one name (e.g. 'Ashbourne' for the
estate and 'Ashburton' for the surname), the patch would write a new contradiction into the script. Both are
held until an editor sets one canonical value; a separate preview shows what one shared spelling would do.

Only names are patched: a trait or injury needs a writer's judgement about which line is wrong.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

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
                         "by_editor": bool(d.get("canonical")),
                         "accepted": d.get("decision") == "accept" or bool(d.get("canonical")),
                         "chosen_by": "editor" if d.get("canonical") else "tool suggestion (not yet accepted by an editor)"})
    # hold flags whose kept spellings are themselves variants of each other, unless an editor chose both
    for i, a in enumerate(plan):
        for b in plan[i + 1:]:
            if a["keep"] != b["keep"] and fuzz.ratio(a["keep"].lower(), b["keep"].lower()) >= 75 \
                    and not (a["by_editor"] and b["by_editor"]):
                for x, y in ((a, b), (b, a)):
                    x.setdefault("held_with", []).append(y["flag"])
    for p in plan:
        if p.get("held_with"):
            p["status"] = (f"HELD: keeps '{p['keep']}' while flag {', '.join(p['held_with'])} keeps a variant of it; "
                           "an editor must choose one spelling for both")
        elif p["accepted"]:
            p["status"] = "applied"
        else:  # human confirms: the tool's suggestion goes to the preview until an editor accepts the flag
            p["status"] = "suggested: accept the flag in review/decisions.yaml to apply it"
    return plan


def _patch(data: Path, plan: list[dict], out: Path) -> dict:
    pdir = out / "patched_scripts"
    pdir.mkdir(parents=True, exist_ok=True)
    diff_lines, changed = [], 0
    for src in sorted(data.glob("*.txt")):
        text = src.read_text(encoding="utf-8")
        new_lines = []
        for ln in text.splitlines(keepends=True):
            new = ln
            for p in plan:
                for o in p["replace"]:
                    new = re.sub(rf"\b{re.escape(o)}\b", p["keep"], new)
            changed += new != ln
            new_lines.append(new)
        patched = "".join(new_lines)
        (pdir / src.name).write_text(patched, encoding="utf-8")
        if patched != text:
            diff_lines += difflib.unified_diff(text.splitlines(keepends=True), new_lines,
                                               f"a/{src.name}", f"b/{src.name}", n=0)
    (out / "fix.diff").write_text("".join(diff_lines), encoding="utf-8")
    return {"lines_changed": changed, "after": variant_groups(parse_folder(pdir))}


def _fmt(groups):
    return [", ".join(f"{m['name']} x{m['count']}" for m in g["members"]) for g in groups]


def apply_fixes(data: Path, flags: list[dict], decisions: dict, out: Path) -> dict:
    plan = plan_fixes(flags, decisions)
    before = variant_groups(parse_folder(data))
    counts = {m["name"]: m["count"] for g in before for m in g["members"]}
    main = _patch(data, [p for p in plan if p["status"] == "applied"], out)
    rec = {"plan": plan, "lines_changed": main["lines_changed"],
           "spelling_groups_before": _fmt(before), "spelling_groups_after": _fmt(main["after"]),
           "patched_scripts": "patched_scripts/ (local only: full scripts)", "diff": "fix.diff"}
    pending = [p for p in plan if p["status"] != "applied"]
    if pending:  # preview, NOT applied: every suggestion; clashing ones share the most frequent of their kept values
        held = [p for p in pending if p.get("held_with")]
        shared = max({p["keep"] for p in held}, key=lambda v: counts.get(v, 0)) if held else None
        prev_plan = [p for p in plan if p["status"] == "applied"] + [
            {**p, "keep": shared, "replace": sorted(set(p["replace"]) | ({p["keep"]} - {shared}))} if p.get("held_with") else p
            for p in pending]
        prev = _patch(data, prev_plan, out / "preview_one_spelling")
        rec["preview_one_spelling"] = {"shared_value": shared or "n/a", "lines_changed": prev["lines_changed"],
                                       "spelling_groups_after": _fmt(prev["after"]),
                                       "note": "NOT applied: what the patch would do if an editor accepted every suggestion"
                                               + (f" and chose '{shared}' for the clashing flags" if shared else "")}
    (out / "fix_and_recheck.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec
