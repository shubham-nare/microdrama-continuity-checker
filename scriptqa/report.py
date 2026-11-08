"""Outputs: series bible, per-scene reference cards, flags (json/csv) and one readable HTML report."""
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

from .parse import Episode, fmt_ts
from .resolve import canon_subject

import re

TRAIT_KINDS = ("TRAIT", "PROP", "INJURY")
STABLE_TRAITS = {"eye_colour", "hair", "facial_hair", "scar", "build", "age", "skin", "voice"}
TEMPORARY_WORDS = {"wet", "soaked", "scratched", "bruised", "bloody", "bloodied", "muddy", "mud", "dirty", "tear",
                   "tears", "sweat", "sweaty", "dishevelled", "disheveled", "tangled", "smudged", "pale", "flushed"}


def _stable(attr: str, value: str) -> bool:
    words = set(value.lower().replace(",", " ").split())
    return attr in STABLE_TRAITS and not (words & TEMPORARY_WORDS)


def gate_replay(eps: list[Episode], flags: list[dict]) -> None:
    """For each contradiction: the episode where a per-episode gate would have blocked it (the first episode
    where a second, different value appears), and what reaches production if it is not caught: later
    dialogue lines and CAMERA shots that carry a non-majority value."""
    owned = {}
    for f in flags:
        if f["type"] == "CONTRADICTION":
            for v in (f.get("variant_counts") or {}):
                owned.setdefault(v.lower(), []).append(f["id"])
    for f in flags:
        if f["type"] != "CONTRADICTION":
            continue
        counts = f.get("variant_counts") or {}
        if not counts:
            continue
        seen, gate = set(), None
        for ev in sorted(f["evidence"], key=lambda e: (e["episode"], e["timestamp"])):
            seen.add(ev["value"])
            if len(seen) >= 2:
                gate = ev["episode"]
                break
        if not gate:
            continue
        # running counts: occurrences of each value in episodes BEFORE the gate (what a gate would know)
        before = {v: sum(len(re.findall(rf"\b{re.escape(v.lower())}\b", el.text.lower()))
                         for ep in eps if ep.number < gate for el in ep.elements) for v in counts}
        major = max(before, key=lambda v: (before[v], -list(counts).index(v)))
        odd = [v for v in counts if v != major and len(v) >= 3 and len(owned.get(v.lower(), [])) <= 1]
        lines = shots = 0
        if gate and odd:
            pat = [o.lower() for o in odd]
            for ep in eps:
                if ep.number < gate:
                    continue
                hit_scenes = set()
                for el in ep.elements:
                    t = el.text.lower()
                    if any(o in t for o in pat):
                        if el.kind == "DIALOGUE":
                            lines += 1
                        hit_scenes.add(el.scene)
                shots += sum(1 for el in ep.elements if el.kind == "CAMERA" and el.scene in hit_scenes)
        f["gate"] = {"blocked_at_episode": gate, "majority_value": major, "odd_values": odd,
                     "later_dialogue_lines": lines, "later_camera_shots_in_affected_scenes": shots}


def build_bible(eps: list[Episode], facts: list[dict], res: dict, name_groups: list[dict]) -> dict:
    chars = {}
    for canon, ch in res["characters"].items():
        chars[canon] = {"id": ch["id"], "canonical": canon, "labels": ch["labels"],
                        "episodes": sorted({e for es in ch["labels"].values() for e in es}),
                        "traits": defaultdict(list), "costumes": [], "props": defaultdict(list),
                        "injuries": [], "relationships": defaultdict(list), "names": defaultdict(list)}
    other = defaultdict(list)
    for f in facts:
        c = f.get("canon") or canon_subject(f["subject"], f["episode"], res)
        ref = {"episode": f["episode"], "element_id": f["element_id"], "value": f["value"], "quote": f["quote"]}
        if c not in chars:
            other[c].append({"kind": f["kind"], "attribute": f["attribute"], **ref})
            continue
        d = chars[c]
        k = f["kind"]
        if k == "TRAIT":
            d["traits"][f["attribute"].lower()].append(ref)
        elif k == "COSTUME":
            d["costumes"].append(ref)
        elif k == "PROP":
            d["props"][f["attribute"].lower()].append(ref)
        elif k == "INJURY":
            d["injuries"].append({"attribute": f["attribute"], **ref})
        elif k == "RELATIONSHIP":
            d["relationships"][f["attribute"].lower()].append(ref)
        elif k == "NAME":
            d["names"][f["attribute"].lower()].append(ref)
    for d in chars.values():
        for key in ("traits", "props", "relationships", "names"):
            d[key] = dict(d[key])
    ordered = dict(sorted(chars.items(), key=lambda kv: -len(kv[1]["episodes"])))
    return {"characters": ordered, "non_character_facts": dict(other),
            "name_spelling_groups": [[m["name"] for m in g["members"]] for g in name_groups]}


def build_cards(eps: list[Episode], facts: list[dict], res: dict, flags: list[dict]) -> list[dict]:
    """Per scene: every character on screen (one canonical ID whatever label the scene uses) with the
    traits/props known UP TO this episode (latest value wins), injuries stated in this or the previous
    episode only (older injuries are treated as healed unless restated), this episode's costume, and the
    open flags touching the scene. Each card is a snapshot: nothing from later episodes leaks in."""
    traits = defaultdict(dict)   # canon -> attribute -> "value (epN)"
    props = defaultdict(dict)    # canon -> object -> (episode, "value")
    injuries = defaultdict(list)  # canon -> [(episode, attribute, value)]
    by_ep = defaultdict(list)
    for f in facts:
        by_ep[f["episode"]].append(f)
    flag_by_el = defaultdict(list)
    for fl in flags:
        if fl["severity"] in ("high", "medium"):
            odd = [e for e in fl["evidence"] if e.get("odd")] or fl["evidence"]
            for e in odd:  # only the lines that need fixing carry the flag, not the majority-spelling lines
                flag_by_el[e["element_id"]].append(fl["id"])
    cards = []
    for ep in eps:
        ep_facts = sorted(by_ep[ep.number], key=lambda f: (bool(f.get("is_change")), f.get("start", 0.0)))
        costume = defaultdict(list)
        for f in ep_facts:
            c = f.get("canon") or canon_subject(f["subject"], f["episode"], res)
            if f["kind"] == "TRAIT":
                traits[c][f["attribute"].lower()] = (ep.number, f["value"])
            elif f["kind"] == "PROP":
                props[c][f["attribute"].lower()] = (ep.number, f["value"])
            elif f["kind"] == "INJURY":
                injuries[c].append((ep.number, f["attribute"].lower(), f["value"]))
            elif f["kind"] == "COSTUME":
                costume[(c, f["scene"])].append(f["value"])
        for sc in ep.scenes or []:
            els = [e for e in ep.elements if e.scene == sc.number]
            present = {}
            for e in els:
                if e.speaker:
                    canon = res["assign"].get(f"{e.speaker}|{ep.number}", e.speaker.title())
                    present.setdefault(canon, set()).add(e.speaker)
            for f in ep_facts:
                if f["scene"] == sc.number and f.get("element_kind") in ("CAMERA", "ACTION"):
                    c = f.get("canon") or canon_subject(f["subject"], f["episode"], res)
                    if c in res["characters"]:
                        present.setdefault(c, set())
            people = []
            for canon, labels in sorted(present.items()):
                active = [f"{v} ({a.replace(' location', '').replace(' cause', ', cause')}, ep{n})"
                          for n, a, v in injuries[canon] if ep.number - 1 <= n <= ep.number]
                recent_props = {o: f"{v} (ep{n})" for o, (n, v) in props[canon].items() if ep.number - n <= 2}
                # stable traits (eyes, hair, scar...) carry forward; temporary ones (wet hair, scratched cheeks) only
                # count in the episode they are stated in and the next one
                live_traits = {a: f"{v} (ep{n})" for a, (n, v) in traits[canon].items()
                               if _stable(a, v) or ep.number - n <= 1}
                people.append({"character": canon, "id": res["characters"].get(canon, {}).get("id", canon.lower()),
                               "labels_in_scene": sorted(labels), "costume_this_scene": list(costume.get((canon, sc.number), [])),
                               "traits": live_traits, "props_recent": recent_props, "injuries": active})
            warn = sorted({fid for e in els for fid in flag_by_el.get(e.id, [])})
            cards.append({"episode": ep.number, "scene": sc.number, "heading": sc.heading,
                          "start": fmt_ts(sc.start or 0), "end": fmt_ts(sc.end or 0),
                          "characters": people, "open_flags": warn})
    return cards


def write_flags_csv(flags: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "id", "status", "severity", "type", "kind", "audible", "summary", "episodes", "suggested_fix",
                    "evidence (episode element timestamp value quote)", "reviewer_decision (dismiss/accept)", "canonical (if accept)"])
        for f in flags:
            ev = " || ".join(f"ep{e['episode']} line {e.get('line', '?')} {e['timestamp']} {'*' if e['odd'] else ''}{e['value']}: \"{e['quote'][:100]}\""
                             for e in f["evidence"])
            w.writerow([f.get("rank"), f["id"], f.get("status", "open"), f["severity"], f["type"], f["kind"], f["audible"], f["summary"],
                        " ".join(map(str, f["episodes"])), f["suggested_fix"], ev, "", ""])


CSS = """body{font-family:system-ui,Segoe UI,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1d1d1f;background:#fff}
h1{font-size:22px}h2{font-size:18px;margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:4px}
.f{border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:10px 0}.high{border-left:6px solid #c62828}
.medium{border-left:6px solid #ef6c00}.low{border-left:6px solid #9e9e9e}.info{border-left:6px solid #1565c0}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:10px;background:#eee;margin-right:4px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #eee;padding:4px 6px;text-align:left;vertical-align:top}
.odd{background:#fff3e0}.q{color:#444;font-style:italic}details{margin:6px 0}small{color:#666}
@media (prefers-color-scheme: dark){body{background:#121212;color:#eee}.f{border-color:#333}td,th{border-color:#2a2a2a}.odd{background:#3a2a10}.tag{background:#333}.q{color:#bbb}}"""


RISK = {("names", "DIFFERENT_ENTITIES"): 0, ("facts", "INTENTIONAL_CHANGE"): 1, ("facts", "DIFFERENT_ENTITIES"): 2,
        ("facts", "COMPATIBLE"): 3}


def cleared_groups(groups: list[dict]) -> list[dict]:
    out = []
    for g in groups:
        v = g.get("verdict", {}).get("verdict")
        if v in ("CONTRADICTION", "UNCLEAR") or (g["source"] == "names" and v in ("INTENTIONAL_CHANGE", "COMPATIBLE")):
            continue  # those became flags
        out.append({"source": g["source"], "kind": g["kind"], "subject": str(g["subject"]), "attribute": str(g["attribute"]),
                    "verdict": v, "reason": g.get("verdict", {}).get("reason", ""),
                    "values": sorted({i["value"] for i in g["items"]})[:8],
                    "episodes": sorted({i["episode"] for i in g["items"]})})
    out.sort(key=lambda c: (RISK.get((c["source"], c["verdict"]), 9), c["kind"]))
    return out


def write_html(path: Path, show: str, flags: list[dict], bible: dict, cards: list[dict], summary: dict,
               cleared: list[dict] | None = None) -> None:
    e = html.escape
    out = [f"<!doctype html><meta charset=utf-8><title>Continuity report — {e(show)}</title><style>{CSS}</style>",
           f"<h1>Continuity report — {e(show)}</h1>",
           (f"<p style='background:#fdecea;padding:8px;border-radius:6px'><b>{len(summary.get('episodes_not_read', []))} episode(s) were NOT read by the model</b> "
            f"(episodes {', '.join(map(str, summary.get('episodes_not_read', [])))}): only the code-only name check covers them. Re-run to retry.</p>"
            if summary.get("episodes_not_read") else ""),
           f"<p>{summary['episodes']} episodes · {summary['elements']} timed elements · "
           f"{summary['facts_verified']} verified facts · <b>{summary['flags_contradiction']} contradictions</b> "
           f"({summary['flags_high']} high) · {summary['flags_collision']} label collisions · {summary['flags_alias']} characters with several labels · "
           f"{summary['flags_review']} marked 'needs a human look' · every flag is open until an editor accepts or dismisses it</p>",
           "<p><small>Every quote below was checked by code against the script. Severity: high = audible (dialogue) or "
           "breaks who-is-who; medium = visual state (trait, prop, injury); info = alias list for reference images. "
           "Reviewer: mark each flag accept / dismiss in flags.csv (column reviewer_decision).</small></p>"]
    sections = [("Contradictions", [f for f in flags if f["type"] == "CONTRADICTION"]),
                ("Speaker-label collisions", [f for f in flags if f["type"] == "LABEL_COLLISION"]),
                ("Characters under several labels", [f for f in flags if f["type"] == "LABEL_ALIAS"]),
                ("Needs a human look", [f for f in flags if f["type"] == "REVIEW"])]
    for title, fl in sections:
        out.append(f"<h2>{e(title)} ({len(fl)})</h2>")
        if title == "Characters under several labels":
            out.append("<p><small>Not errors: the label map that reference images and voices should use (one character ID per row).</small></p>"
                       "<table><tr><th>character</th><th>labels (episodes)</th><th>id</th></tr>")
            for f in fl:
                labs = "; ".join(f"{ev['value']} (ep{ev['episode']}…)" for ev in f["evidence"])
                out.append(f"<tr><td>{e(f['subject'])}</td><td>{e(labs)}</td><td>{e(f['suggested_fix'].split(chr(39))[1] if chr(39) in f['suggested_fix'] else '')}</td></tr>")
            out.append("</table>")
            continue
        for f in fl:
            out.append(f"<div class='f {f['severity']}'><b>#{f['rank']} {e(f['summary'])}</b><br>"
                       f"<span class=tag>{f['severity']}</span><span class=tag>{f['kind']}</span>"
                       + ("<span class=tag>audible</span>" if f["audible"] else "")
                       + f"<span class=tag>{e(f['id'])}</span> <small>{e(f['suggested_fix'])} — {e(f['adjudication'].get('reason', ''))}</small>"
                       + (f"<br><small><b>A per-episode check (run.py --gate) would have stopped this at ep{f['gate']['blocked_at_episode']}.</b> Not caught, the minority value(s) {e(', '.join(f['gate']['odd_values'][:5]))} reach "
                          f"{f['gate']['later_dialogue_lines']} later spoken line(s).</small>"
                          if f.get("gate", {}).get("blocked_at_episode") else "")
                       + "<table><tr><th>ep</th><th>line</th><th>time</th><th>where</th><th>value</th><th>quote</th></tr>")
            for ev in f["evidence"]:
                out.append(f"<tr class='{'odd' if ev['odd'] else ''}'><td>{ev['episode']}</td><td>{ev.get('line', '')}</td><td>{ev['timestamp']}</td>"
                           f"<td>{e(str(ev['where']))}</td><td>{e(ev['value'])}</td><td class=q>“{e(ev['quote'][:200])}”</td></tr>")
            out.append("</table></div>")
    if cleared is not None:
        out.append(f"<h2>Cleared by the model ({len(cleared)})</h2><p><small>Candidates the adjudicator judged compatible, intentional or "
                   "about different things. They are not flags, but nothing is hidden: riskiest first. To overrule one, add an accept "
                   "line for it in review/decisions.yaml or tell the team.</small></p><details><summary>show all</summary><table>"
                   "<tr><th>kind</th><th>subject / attribute</th><th>values</th><th>eps</th><th>model said</th></tr>")
        for c in cleared:
            out.append(f"<tr><td>{e(c['kind'])}</td><td>{e(c['subject'])} — {e(c['attribute'])}</td><td>{e('; '.join(c['values']))}</td>"
                       f"<td>{e(', '.join(map(str, c['episodes'][:8])))}</td><td><b>{e(c['verdict'] or '')}</b>: {e(c['reason'][:160])}</td></tr>")
        out.append("</table></details>")
    out.append("<h2>Series bible — main characters</h2>")
    for canon, d in list(bible["characters"].items())[:12]:
        out.append(f"<details><summary><b>{e(canon)}</b> <small>id {e(d['id'])} · eps {d['episodes'][0] if d['episodes'] else ''}–{d['episodes'][-1] if d['episodes'] else ''} · labels: {e(', '.join(d['labels']))}</small></summary><table>")
        for attr, refs in d["traits"].items():
            out.append(f"<tr><td>trait: {e(attr)}</td><td>{e('; '.join(sorted({r['value'] + ' (ep' + str(r['episode']) + ')' for r in refs})))}</td></tr>")
        for attr, refs in d["props"].items():
            out.append(f"<tr><td>prop: {e(attr)}</td><td>{e('; '.join(r['value'] + ' (ep' + str(r['episode']) + ')' for r in refs[:8]))}</td></tr>")
        for r in d["injuries"][:10]:
            out.append(f"<tr><td>injury: {e(r['attribute'])}</td><td>{e(r['value'])} (ep{r['episode']})</td></tr>")
        for attr, refs in d["relationships"].items():
            out.append(f"<tr><td>{e(attr)}</td><td>{e('; '.join(sorted({r['value'] + ' (ep' + str(r['episode']) + ')' for r in refs})))}</td></tr>")
        out.append("</table></details>")
    out.append(f"<h2>Reference cards (per scene, for shot prompts)</h2><p><small>{len(cards)} scenes — full list in reference_cards.json. First 6 shown.</small></p>")
    for c in cards[:6]:
        out.append(f"<details><summary>ep{c['episode']} scene {c['scene']} — {e(c['heading'])}</summary><table>")
        for p in c["characters"]:
            out.append(f"<tr><td><b>{e(p['character'])}</b><br><small>labels: {e(', '.join(p['labels_in_scene']))}</small></td>"
                       f"<td>costume: {e('; '.join(p['costume_this_scene']) or 'not stated')}<br>traits: {e(json.dumps(p['traits'], ensure_ascii=False))}"
                       f"<br>props (last 3 eps): {e(json.dumps(p['props_recent'], ensure_ascii=False))}<br>injuries: {e('; '.join(p['injuries']) or 'none')}</td></tr>")
        out.append("</table></details>")
    path.write_text("\n".join(out), encoding="utf-8")
