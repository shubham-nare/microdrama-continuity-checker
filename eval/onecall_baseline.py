"""The obvious build, measured: the whole show in ONE long-context call, "list every continuity error".

Scored with the same answer key and the same code checks as the pipeline:
  - which answer-key errors it finds (eval/evaluate.py: score_real / match_gold)
  - how many of its cited quotes actually exist in the script (scriptqa/verify.py: verify_item)
  - how many of its items match nothing in the key (unreviewed; reported, not labelled)

usage: python -I eval/onecall_baseline.py --data data --out outputs/onecall
The answer is cached like every other call, so a re-run is free and identical.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from scriptqa.llm import LLM, LLMError  # noqa: E402
from scriptqa.parse import parse_folder, render_for_model  # noqa: E402
from scriptqa.verify import verify_item  # noqa: E402


class Ev(BaseModel):
    episode: int
    element_id: str = Field(description="the [E...] id of the line")
    quote: str = Field(description="exact words copied from that line")
    value: str = Field(description="the conflicting value as written in this line, e.g. a name spelling, a colour, a body part")


class Err(BaseModel):
    kind: Literal["NAME", "TRAIT", "COSTUME", "PROP", "INJURY", "RELATIONSHIP", "LABEL_COLLISION", "OTHER"]
    subject: str = Field(description="who or what; for LABEL_COLLISION the speaker label exactly as written")
    description: str
    evidence: list[Ev] = Field(description="at least two lines that conflict")


class Errs(BaseModel):
    errors: list[Err]


SYSTEM = ("You are a script supervisor checking an AI-written vertical drama series for continuity errors before video "
          "generation. Each line of the script is tagged with an id in square brackets.")
PROMPT_HEAD = ("Below is the full series. List every continuity error: things that contradict each other across or within "
               "episodes with no on-screen reason (names spelled differently, a trait or injury that moves or changes, a prop "
               "that changes, contradictory family relationships, one speaker label used for two different people). Do not "
               "list intentional story changes. For each error give at least two evidence lines with the line id, an exact "
               "quote copied from that line, and the conflicting value.\n\n")


EXHAUSTIVE = ("Be exhaustive: go through every character name and surname, every place and title name, every physical trait, "
              "every injury and where it is on the body, every important prop, every family relationship and every speaker "
              "label, and report each inconsistency separately.\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "onecall")
    ap.add_argument("--exhaustive", action="store_true", help="prompt variant asking it to check every name, trait, injury, prop and relationship")
    a = ap.parse_args()
    os.environ.setdefault("LLM_MAX_TOKENS", "32000")
    eps = parse_folder(a.data)
    by_num = {e.number: e for e in eps}
    prompt = PROMPT_HEAD + (EXHAUSTIVE if a.exhaustive else "") + "\n\n".join(render_for_model(e) for e in eps)
    llm = LLM()
    t0 = time.time()
    try:
        res = llm.complete_json(SYSTEM, prompt, Errs, tag="onecall")
    except LLMError as e:
        print(f"[onecall] FAILED: {e}")
        return 1
    secs = round(time.time() - t0, 1)

    flags, n_q, q_ok, q_rep, q_noval = [], 0, 0, 0, 0
    for i, er in enumerate(res["errors"]):
        evs = []
        for ev in er["evidence"]:
            n_q += 1
            ep = by_num.get(ev["episode"])
            # fair to the model: it may copy the rendered prefix ("ACTION: ", "NAME (emotion): ") and the dialogue quotes
            q = re.sub(r"^\s*(\[E\d+-\d+\]\s*)?(\d\d:\d\d\.\d\d\s*)?[A-Z][A-Z'’ .\-]*(\s*\([^)]*\))?:\s*", "", ev["quote"]).strip().strip('"')
            st = verify_item({**ev, "quote": q}, ep)[0] if ep else "dropped"
            if st != "dropped" and ev["value"].lower() not in ev["quote"].lower():
                st = "value_not_in_quote"  # the line exists but does not contain the value it is cited for
            q_ok += st == "ok"
            q_rep += st == "repointed"
            q_noval += st == "value_not_in_quote"
            evs.append({**ev, "verify": st})
        ftype = "LABEL_COLLISION" if er["kind"] == "LABEL_COLLISION" else "CONTRADICTION"
        flags.append({"id": f"OC-{i:03d}", "type": ftype, "kind": er["kind"], "subject": er["subject"],
                      "summary": er["description"], "episodes": sorted({e["episode"] for e in evs}), "evidence": evs})

    import evaluate as ev_mod  # eval/evaluate.py: same answer key, same matching rule
    found = {g["id"]: [f["id"] for f in flags if ev_mod.match_gold(f, g)]
             for g in (ev_mod._gold("gold_real.yaml") or {}).get("items", [])}
    # strict variant: only evidence whose quote really is in the script counts
    strict_flags = [{**f, "evidence": [e for e in f["evidence"] if e["verify"] in ("ok", "repointed")]} for f in flags]
    strict_flags = [f for f in strict_flags if len(f["evidence"]) >= 2]
    found_strict = {g: [f["id"] for f in strict_flags if ev_mod.match_gold(f, gg)]
                    for g, gg in ((g["id"], g) for g in (ev_mod._gold("gold_real.yaml") or {}).get("items", []))}
    matched = {fid for v in found.values() for fid in v}
    u = llm.usage()
    rec = {
        "approach": "one long-context call over the whole show" + (" (exhaustive prompt)" if a.exhaustive else ""),
        "model": u["models_used"], "calls_live": u["calls_live"], "in_tokens": u["in_tokens"], "out_tokens": u["out_tokens"],
        "seconds_this_run": secs,
        "n_items": len(flags),
        "gold_found": sum(1 for v in found.values() if v), "gold_total": len(found), "found": found,
        "gold_found_with_verified_quotes": sum(1 for v in found_strict.values() if v),
        "quotes": {"total": n_q, "on_cited_line": q_ok, "elsewhere_in_episode": q_rep, "line_lacks_the_value": q_noval,
                   "not_in_script": n_q - q_ok - q_rep - q_noval},
        "items_matching_no_key_entry": len(flags) - len(matched),
        "flags": flags,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "onecall.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[onecall] {rec['n_items']} items; answer key {rec['gold_found']}/{rec['gold_total']} "
          f"({rec['gold_found_with_verified_quotes']} with real quotes); quotes on the cited line {q_ok}/{n_q}, "
          f"line lacks the value {q_noval}, not in the script {n_q - q_ok - q_rep - q_noval}; {u['calls_live']} live call(s), {u['in_tokens']}+{u['out_tokens']} tokens, {secs}s")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "eval"))
    raise SystemExit(main())
