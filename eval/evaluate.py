"""Evaluation + deterministic checks for the continuity checker (all code, no model judgement).

  1. seeded-error test: copy the scripts, inject known errors with a fixed seed, re-run, measure recall
     per error type for the pipeline AND for the code-only baseline (names.py)
  2. real-pack recall against eval/gold_real.yaml (hand-verified real errors)
  3. real-pack precision: every non-info flag is a gold match or has a label in eval/flag_labels.yaml
  4. alias accuracy against eval/gold_aliases.yaml (pairwise precision/recall over label-episode pairs)
  5. deterministic checks 1-10 from state/DECISIONS.md, including the model-swap test
Writes eval.json + checks.json into the run folder and prints a summary. Returns True if all checks pass.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scriptqa.llm import LLM  # noqa: E402
from scriptqa.names import variant_groups  # noqa: E402
from scriptqa.parse import parse_folder, TS  # noqa: E402
from scriptqa.pipeline import run as run_pipeline  # noqa: E402
from scriptqa.verify import norm  # noqa: E402

SEED = 7
NON_INFO = ("CONTRADICTION", "REVIEW", "LABEL_COLLISION")
SCORED = NON_INFO + ("LABEL_ALIAS",)

# --------------------------------------------------------------------------- seeded errors
VOWEL_SWAP = {"a": "e", "e": "a", "i": "e", "o": "a", "u": "o"}


def name_variants(eps, top: int = 8) -> dict[str, str]:
    """Derive realistic misspellings of the show's own most frequent names (one vowel swapped after
    the first two letters), so no show-specific names live in the code."""
    from scriptqa.names import name_inventory
    inv = name_inventory(eps)
    names = [w for w in sorted(inv, key=lambda w: (-len(inv[w]), w)) if len(inv[w]) >= 5 and len(w) >= 5][:top]
    out = {}
    for n in names:
        for i in range(2, len(n)):
            if n[i] in VOWEL_SWAP:
                out[n] = n[:i] + VOWEL_SWAP[n[i]] + n[i + 1:]
                break
    return out
COLOURS = {"blue": "green", "green": "brown", "grey": "brown", "gray": "brown", "brown": "blue", "hazel": "blue"}
MATERIALS = {"silver": "gold", "gold": "silver", "emerald": "ruby", "amethyst": "sapphire", "brass": "iron"}
RELATIONS = {"cousin": "brother", "mother": "aunt", "father": "uncle", "sister": "cousin", "brother": "cousin",
             "husband": "brother", "wife": "sister", "son": "nephew", "daughter": "niece"}


def find_sites(eps) -> dict[str, list[dict]]:
    sites = defaultdict(list)
    NAME_VARIANTS = name_variants(eps)
    for ep in eps:
        for el in ep.elements:
            t = el.text
            if el.kind == "DIALOGUE":
                for n, v in NAME_VARIANTS.items():
                    if re.search(rf"\b{n}\b", t):
                        sites["name_variant"].append({"el": el, "old": n, "new": v})
                m = re.search(r"\b(my|your|his|her)\s+(cousin|mother|father|sister|brother|husband|wife|son|daughter)\b", t, re.I)
                if m:
                    sites["relation_flip"].append({"el": el, "old": m.group(2), "new": RELATIONS[m.group(2).lower()],
                                                   "ctx": m.group(0)})
            m = re.search(r"\b(blue|green|grey|gray|brown|hazel)(?:-\w+)?\s+(eyes|gaze)\b", t, re.I)
            if m:
                sites["eye_colour"].append({"el": el, "old": m.group(1), "new": COLOURS[m.group(1).lower()], "ctx": m.group(0)})
            m = re.search(r"\b(left|right)\s+(shoulder|hand|arm|cheek|eye|leg|palm|temple|side|brow|wrist)\b", t, re.I)
            # only a side swap on an INJURY is a continuity error (a gesture with the left hand is not)
            if m and re.search(r"\b(scar|scarred|wound|wounded|bandage|bandaged|bruise|bruised|cut|burn|burned|welt|bleeding|blood|injur\w*|gash)\b", t, re.I):
                sites["injury_side"].append({"el": el, "old": m.group(1), "new": "right" if m.group(1).lower() == "left" else "left",
                                             "ctx": m.group(0)})
            m = re.search(r"\b(silver|gold|emerald|amethyst|brass)\b[\w\s,-]{0,25}\b(ring|locket|pendant|cane|band|watch)\b", t, re.I)
            if m:
                sites["prop_material"].append({"el": el, "old": m.group(1), "new": MATERIALS[m.group(1).lower()], "ctx": m.group(0)})
    return sites


def validity_class(eps, m: dict) -> str:
    """cross-episode: the old value appears with the same key word in ANOTHER episode (so the plant contradicts
    something elsewhere); same-episode: only elsewhere in its own episode; none: nothing to contradict."""
    key = (m.get("ctx") or "").split()[-1].lower() if m.get("ctx") else None
    old = m["old"].lower()
    other = same = False
    for ep in eps:
        for el in ep.elements:
            if el.id == m["element_id"]:
                continue
            t = el.text.lower()
            if re.search(rf"\b{re.escape(old)}\b", t) and (key is None or key in t):
                if ep.number == m["episode"]:
                    same = True
                else:
                    other = True
    return "cross-episode" if other else ("same-episode" if same else "none")


def make_mutations(eps, per_type: int = 4, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    sites = find_sites(eps)
    used_eps, muts = set(), []
    for mtype in sorted(sites):
        cand = sites[mtype][:]
        rng.shuffle(cand)
        n = 0
        for s in cand:
            if s["el"].episode in used_eps or s["el"].episode <= 2:
                continue
            used_eps.add(s["el"].episode)
            el = s["el"]
            muts.append({"type": mtype, "episode": el.episode, "element_id": el.id, "line": el.line,
                         "old": s["old"], "new": _match_case(s["new"], s["old"]), "before": el.text,
                         "ctx": s.get("ctx", "")})
            n += 1
            if n >= per_type:
                break
    return muts


def _match_case(new: str, old: str) -> str:
    return new.capitalize() if old[:1].isupper() else new


def apply_mutations(src: Path, dst: Path, eps, muts: list[dict]) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    by_ep = {e.number: e for e in eps}
    for p in src.glob("*.txt"):
        shutil.copy(p, dst / p.name)
    for m in muts:
        ep = by_ep[m["episode"]]
        path = dst / ep.file
        lines = path.read_text(encoding="utf-8").splitlines()
        # element text starts 2 lines after the label (label, timestamps, text)
        for k in range(m["line"] - 1, min(m["line"] + 3, len(lines))):  # m["line"] is the 1-based text line
            if re.search(rf"\b{re.escape(m['old'])}\b", lines[k]):
                lines[k] = re.sub(rf"\b{re.escape(m['old'])}\b", m["new"], lines[k], count=1)
                m["after_line"] = lines[k].strip()
                break
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flag_hits_mutation(f: dict, m: dict) -> bool:
    if f["type"] not in NON_INFO:
        return False
    for e in f["evidence"]:
        if e["element_id"] == m["element_id"]:
            return True
        if e["episode"] == m["episode"] and norm(m["new"]) in norm(e["value"] + " " + e["quote"]):
            return True
    return False


def baseline_detects(groups: list[dict], m: dict) -> bool:
    for g in groups:
        for mem in g["members"]:
            if mem["name"].lower() == m["new"].lower() and m["episode"] in mem["episodes"]:
                return True
    return False


# --------------------------------------------------------------------------- gold scoring
def flag_text(f: dict) -> str:
    return norm(" ".join(e["value"] + " " + e["quote"] for e in f["evidence"]) + " " + f["summary"])


def match_gold(f: dict, g: dict) -> bool:
    if g.get("flag_type"):
        return f["type"] == g["flag_type"] and f["subject"].upper() == g["label"].upper()
    if f["type"] not in ("CONTRADICTION", "REVIEW"):
        return False
    txt = flag_text(f)
    hits = sum(1 for v in g["values"] if re.search(rf"\b{re.escape(norm(v))}\b", txt))
    if hits < g.get("min_values", 2):
        return False
    eps = set(f["episodes"])
    if g.get("episodes_any") and not eps & set(g["episodes_any"]):
        return False
    if g.get("episodes_all") and not set(g["episodes_all"]) <= eps:
        return False
    return True


GOLD_DIR = ROOT / "eval"


def _gold(name: str):
    p = GOLD_DIR / name
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None


def judge_alias_flag(f: dict) -> dict:
    """A 'character under several labels' flag is true if every label it lists that the gold file knows
    belongs to the same gold character (in those episodes), false if it merges different gold characters."""
    gold = (_gold("gold_aliases.yaml") or {}).get("characters", {})
    owner = defaultdict(set)
    for char, members in gold.items():
        for m in members:
            owner[m["label"]].add((char, "all" if m["episodes"] == "all" else tuple(m["episodes"])))
    chars = set()
    for ev in f["evidence"]:
        for char, eps in owner.get(ev["value"], ()):
            if eps == "all" or ev["episode"] in eps:
                chars.add(char)
    known = [ev["value"] for ev in f["evidence"] if ev["value"] in owner]
    if len(chars) > 1:
        return {"label": "false", "why": f"merges different gold characters {sorted(chars)}"}
    if len(known) >= 2 and len(chars) == 1:
        return {"label": "true", "why": f"all labels are gold {next(iter(chars))}"}
    return {"label": "unlabelled", "summary": f["summary"]}


def loss_stages(result: dict) -> dict:
    """For each gold item: the pipeline stage where it was lost (code only, from facts, groups and flags)."""
    gold = (_gold("gold_real.yaml") or {}).get("items", [])
    out = {}
    for g in gold:
        if any(match_gold(f, g) for f in result["flags"]):
            out[g["id"]] = "flagged"
            continue
        if g.get("flag_type"):
            out[g["id"]] = "missed by alias resolution"
            continue
        pseudo = lambda items: {"type": "CONTRADICTION", "summary": "", "episodes": sorted({i["episode"] for i in items}),
                                "evidence": [{"value": i["value"], "quote": i["quote"]} for i in items]}
        grp = next((x for x in result["groups"] if match_gold(pseudo(x["items"]), g)), None)
        if grp:
            out[g["id"]] = f"grouped, then cleared by the model as {grp.get('verdict', {}).get('verdict')}: {grp.get('verdict', {}).get('reason', '')[:120]}"
            continue
        about = [f for f in result["facts"] if not g.get("subjects") or
                 any(sj in (f["subject"] + " " + f["attribute"]).lower() for sj in g["subjects"])]
        if match_gold(pseudo(about), g):
            out[g["id"]] = "extracted as facts, but never grouped together (different subject/attribute keys)"
        else:
            out[g["id"]] = "not extracted by the model"
    return out


def score_real(flags: list[dict]) -> dict:
    gold = (_gold("gold_real.yaml") or {}).get("items", [])
    lab_path = GOLD_DIR / "flag_labels.yaml"
    labels = (yaml.safe_load(lab_path.read_text(encoding="utf-8")) or {}).get("labels", {}) if lab_path.exists() else {}
    found = {g["id"]: [f["id"] for f in flags if match_gold(f, g)] for g in gold}
    # strictness: for gold items with a value list, how many of those values appear across the matching flags
    coverage = {}
    for g in gold:
        if g.get("values") and found[g["id"]]:
            txt = " ".join(flag_text(f) for f in flags if f["id"] in found[g["id"]])
            hit = [v for v in g["values"] if re.search(rf"\b{re.escape(norm(v))}\b", txt)]
            coverage[g["id"]] = {"found": len(hit), "of": len(g["values"]), "missing": [v for v in g["values"] if v not in hit]}
    judged = []
    for f in flags:
        if f["type"] not in SCORED:
            continue
        gids = [g["id"] for g in gold if match_gold(f, g)]
        if f["type"] == "LABEL_ALIAS":
            v = judge_alias_flag(f)
            if v["label"] == "unlabelled" and f["id"] in labels:
                v = labels[f["id"]]
            judged.append({"id": f["id"], "type": f["type"], **v})
            continue
        if gids:
            judged.append({"id": f["id"], "type": f["type"], "label": "true", "why": f"matches gold {gids}"})
        elif f["id"] in labels:
            judged.append({"id": f["id"], "type": f["type"], **labels[f["id"]]})
        else:
            judged.append({"id": f["id"], "type": f["type"], "label": "unlabelled", "summary": f["summary"]})
    act = [j for j in judged if j.get("type") != "LABEL_ALIAS"]
    ali = [j for j in judged if j.get("type") == "LABEL_ALIAS"]
    n_true = sum(1 for j in act if j["label"] == "true")
    n_lab = sum(1 for j in act if j["label"] != "unlabelled")
    a_true = sum(1 for j in ali if j["label"] == "true")
    a_lab = sum(1 for j in ali if j["label"] != "unlabelled")
    return {"recall": round(sum(1 for v in found.values() if v) / len(gold), 3) if gold else None, "found": found,
            "coverage": coverage,
            "precision": round(n_true / n_lab, 3) if n_lab else None, "judged": judged,
            "n_flags_scored": len(act), "n_unlabelled": len(act) - n_lab,
            "alias_flag_precision": round(a_true / a_lab, 3) if a_lab else None, "n_alias_flags": len(ali),
            "n_alias_unlabelled": len(ali) - a_lab,
            "distinct_gold_found": sum(1 for v in found.values() if v), "n_gold": len(gold)}


def score_aliases(res_assign: dict) -> dict:
    gold = (_gold("gold_aliases.yaml") or {}).get("characters", {})
    pred = {}
    for key, canon in res_assign.items():
        lab, ep = key.rsplit("|", 1)
        pred[(lab, int(ep))] = canon
    label_eps = defaultdict(set)
    for (lab, ep) in pred:
        label_eps[lab].add(ep)
    truth = {}
    for char, members in gold.items():
        for m in members:
            eps = label_eps.get(m["label"], set()) if m["episodes"] == "all" else set(m["episodes"])
            for e in eps:
                if (m["label"], e) in pred:
                    truth[(m["label"], e)] = char
    pairs = list(itertools.combinations(sorted(truth), 2))
    tp = fp = fn = 0
    errors = []
    for a, b in pairs:
        same_t = truth[a] == truth[b]
        same_p = pred[a] == pred[b]
        tp += same_t and same_p
        fp += (not same_t) and same_p
        fn += same_t and (not same_p)
    # label-level error list (one line per wrong label group)
    seen = set()
    for (lab, e), char in truth.items():
        group_pred = {pred[k] for k in truth if truth[k] == char}
        if len(group_pred) > 1 and (char, lab) not in seen:
            seen.add((char, lab))
            errors.append(f"{char}: label {lab} ep{e} -> predicted '{pred[(lab, e)]}'")
    wrong_merge = sorted({f"{pred[a]} merges {truth[a]} + {truth[b]}" for a, b in pairs if truth[a] != truth[b] and pred[a] == pred[b]})
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    per_char = {}
    multi = [c for c, ms in gold.items() if len(ms) >= 2]
    for char in multi:
        keys = [k for k in truth if truth[k] == char]
        preds = {pred[k] for k in keys}
        others = {pred[k] for k in truth if truth[k] != char}
        per_char[char] = "fully merged" if keys and len(preds) == 1 and not (preds & others) else \
            ("split across " + ", ".join(sorted(preds)) if len(preds) > 1 else "merged with another character")
    n_ok = sum(1 for v in per_char.values() if v == "fully merged")
    return {"pairwise_precision": round(p, 3), "pairwise_recall": round(r, 3),
            "characters_fully_merged": n_ok, "characters_with_several_labels": len(multi), "per_character": per_char,
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, "n_label_episodes": len(truth),
            "split_errors": sorted(errors)[:20], "wrong_merges": wrong_merge[:20]}


# --------------------------------------------------------------------------- checks
def check_quotes(flags, eps) -> tuple[bool, str]:
    by_ep = {e.number: e for e in eps}
    bad, n = [], 0
    for f in flags:
        if f["type"] not in NON_INFO:
            continue
        if len(f["evidence"]) < 2:
            bad.append(f"{f['id']} has {len(f['evidence'])} evidence")
        for ev in f["evidence"]:
            n += 1
            ep = by_ep[ev["episode"]]
            el = ep.by_id().get(ev["element_id"])
            hay = el.text if el else ep.logline + " " + ep.continuity_notes
            if norm(ev["quote"]) not in norm(hay):
                bad.append(f"{f['id']} {ev['element_id']}")
    return not bad, f"{n} evidence quotes checked, {len(bad)} failures {bad[:3]}"


def check_facts(facts, eps) -> tuple[bool, str]:
    by_ep = {e.number: e for e in eps}
    bad = [f for f in facts if norm(f["quote"]) not in norm(by_ep[f["episode"]].by_id()[f["element_id"]].text + " " + (by_ep[f["episode"]].by_id()[f["element_id"]].speaker or ""))]
    return not bad, f"{len(facts)} kept facts re-verified, {len(bad)} failures"


def check_parser(data: Path, eps) -> tuple[bool, str]:
    msgs = []
    raw = 0
    for p in data.glob("*.txt"):
        raw += sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if TS.search(ln))
    parsed = sum(len(e.elements) for e in eps)
    ok = raw == parsed
    msgs.append(f"elements parsed {parsed} vs timestamp lines {raw}")
    by = {e.number: e for e in eps}
    if 6 in by:
        ok6 = len(by[6].scenes) >= 2 and all(s.heading for s in by[6].scenes)
        ok &= ok6
        msgs.append(f"ep6 upper-case headers {'ok' if ok6 else 'FAIL'}")
    if 12 in by:
        starts = [e.start for e in by[12].elements]
        ooo = any(b < a - 0.01 for a, b in zip(starts, starts[1:]))
        msgs.append(f"ep12 out-of-order element kept {'ok' if ooo else 'not present'}")
    if 22 in by:
        past = any(sc.end is not None and e.scene == sc.number and e.end > sc.end for sc in by[22].scenes for e in by[22].elements)
        msgs.append(f"ep22 element past scene end kept {'ok' if past else 'not present'}")
    return ok, "; ".join(msgs)


def swap_test(data: Path, out: Path) -> tuple[bool, str]:
    """Held-out episodes, different provider adapter AND different model, output must validate."""
    from scriptqa.parse import parse_folder as pf
    held = out / "_swap_input"
    if held.exists():
        shutil.rmtree(held)
    held.mkdir(parents=True)
    files = sorted(data.glob("*.txt"))
    for p in [files[i] for i in (0, len(files) // 2, len(files) - 1)]:
        shutil.copy(p, held / p.name)
    env_backup = {k: os.environ.get(k) for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_FALLBACK_MODEL")}
    try:
        os.environ["LLM_PROVIDER"] = os.environ.get("SWAP_PROVIDER", "openai_compat")
        os.environ["LLM_MODEL"] = os.environ.get("SWAP_MODEL", "gemini-3.1-flash-lite")
        os.environ["LLM_BASE_URL"] = os.environ.get("SWAP_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
        os.environ["LLM_API_KEY"] = os.environ.get("SWAP_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        os.environ.pop("LLM_FALLBACK_MODEL", None)
        llm = LLM()
        r = run_pipeline(held, out / "swap_test", "swap-test", None, 3, llm=llm, log=lambda *a: None)
        canary = ""
        demo = ROOT / "demo" / "show"
        if (ROOT / "demo" / "gold" / "gold_real.yaml").exists() and demo.exists():
            rd = run_pipeline(demo, out / "swap_test_demo", "swap-canary", None, 3, llm=LLM(), log=lambda *a: None)
            demo_gold = yaml.safe_load((ROOT / "demo" / "gold" / "gold_real.yaml").read_text(encoding="utf-8"))["items"]
            hits = [g["id"] for g in demo_gold if any(match_gold(f, g) for f in rd["flags"])]
            canary = f"; canary: swapped model catches {len(hits)}/{len(demo_gold)} planted errors in the demo show ({', '.join(hits)})"
        flags = r["flags"]
        ok_schema = all({"id", "type", "severity", "evidence", "summary"} <= set(f) for f in flags)
        ok_q, qmsg = check_quotes(flags, pf(held))
        u = r["runlog"]["usage"]
        ok = ok_schema and ok_q and u["provider"] == "openai_compat" and r["summary"]["facts_verified"] > 0
        return ok, (f"provider={u['provider']} model={u['model']} on {len(list(held.glob('*.txt')))} held-out eps: "
                    f"{r['summary']['facts_verified']} facts, {len(flags)} flags, schema {'ok' if ok_schema else 'FAIL'}, {qmsg}{canary}")
    except Exception as e:
        return False, f"swap failed: {e}"
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(held, ignore_errors=True)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for k successes out of n (honest error bars on small samples)."""
    if n == 0:
        return [0.0, 0.0]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 2), round(min(1.0, c + h), 2)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- main
def evaluate_all(data: Path, out: Path, result: dict, gold_dir: Path | None = None) -> bool:
    global GOLD_DIR
    if gold_dir:
        GOLD_DIR = Path(gold_dir)
    if not (GOLD_DIR / "gold_real.yaml").exists():
        print(f"[eval] no gold files in {GOLD_DIR}: real-pack recall/precision and alias checks will FAIL (not skipped)")
    eps = parse_folder(data)
    flags = result["flags"]
    ev = {}

    # (2)(3) real pack
    real = score_real(flags)
    real["ci95"] = wilson(real.get("distinct_gold_found", 0), real.get("n_gold", 0))
    real["loss_stage"] = loss_stages(result)
    ev["real"] = real
    # (4) aliases
    ev["aliases"] = score_aliases(result["res"].get("model_assign", result["res"]["assign"]))
    ev["aliases_after_review"] = score_aliases(result["res"]["assign"])

    # (1) seeded errors
    muts = make_mutations(eps)
    mdir = ROOT / ".cache" / "mutated_data"
    apply_mutations(data, mdir, eps, muts)
    mres = run_pipeline(mdir, out / "_mutated_run", "seeded-errors", ROOT / "review" / "aliases.yaml", 3,
                        log=lambda *a: None)
    mflags = mres["flags"]
    base_groups = variant_groups(parse_folder(mdir))
    per_type = defaultdict(lambda: {"n": 0, "pipeline": 0, "baseline": 0})
    for m in muts:
        m["pipeline_hit"] = [f["id"] for f in mflags if flag_hits_mutation(f, m)]
        m["baseline_hit"] = baseline_detects(base_groups, m)
        t = per_type[m["type"]]
        t["n"] += 1
        t["pipeline"] += bool(m["pipeline_hit"])
        t["baseline"] += bool(m["baseline_hit"])
        m.pop("before", None)
    clean_ids = {f["id"] for f in flags}
    spurious = [f["id"] for f in mflags if f["type"] in NON_INFO and f["id"] not in clean_ids
                and not any(flag_hits_mutation(f, m) for m in muts)]
    n = len(muts)
    for m in muts:
        m["validity"] = validity_class(eps, m)
        if m["pipeline_hit"]:
            m["loss_stage"] = "flagged"
        elif any(it.get("element_id") == m["element_id"] for g_ in mres["groups"] for it in g_["items"]):
            g_ = next(g_ for g_ in mres["groups"] if any(it.get("element_id") == m["element_id"] for it in g_["items"]))
            m["loss_stage"] = f"grouped, cleared by the model as {g_.get('verdict', {}).get('verdict')}"
        elif any(f_.get("element_id") == m["element_id"] for f_ in mres["facts"]):
            m["loss_stage"] = "extracted, never grouped"
        else:
            m["loss_stage"] = "not extracted"
    by_class = defaultdict(lambda: {"n": 0, "caught": 0})
    for m in muts:
        by_class[m["validity"]]["n"] += 1
        by_class[m["validity"]]["caught"] += bool(m["pipeline_hit"])
    caught = sum(bool(m["pipeline_hit"]) for m in muts)
    distinct = sorted({fid for m in muts for fid in m["pipeline_hit"]})
    ev["seeded"] = {"seed": SEED, "n": n, "caught": caught, "ci95": wilson(caught, n),
                    "by_validity": dict(by_class),
                    "loss_stages": dict(Counter(m["loss_stage"].split(" as ")[0] for m in muts)),
                    "distinct_detecting_flags": len(distinct),
                    "baseline_note": "baseline = code-only spelling detector; it cannot catch non-spelling error types by design",
                    "pipeline_recall": round(sum(bool(m["pipeline_hit"]) for m in muts) / n, 3) if n else None,
                    "baseline_recall": round(sum(bool(m["baseline_hit"]) for m in muts) / n, 3) if n else None,
                    "per_type": dict(per_type), "spurious_new_flags": spurious, "mutations": muts}

    # checks
    checks = []
    def add(i, name, ok, detail):
        checks.append({"n": i, "check": name, "pass": bool(ok), "detail": detail})

    add(1, "every flag has >=2 evidence quotes, each found verbatim at the cited element", *check_quotes(flags, eps))
    add(2, "every kept fact's quote verified; drop rate reported", check_facts(result["facts"], eps)[0],
        check_facts(result["facts"], eps)[1] + f"; dropped at verify: {result['summary']['facts_dropped']}")
    add(3, "parser element counts match + known quirks handled", *check_parser(data, eps))
    a = ev["aliases"]
    coll_ids = [g["id"] for g in ((_gold("gold_real.yaml") or {}).get("items", [])) if g.get("flag_type") == "LABEL_COLLISION"]
    # model-only: does the model's own assignment give each gold collision label to more than one character?
    gold_items = (_gold("gold_real.yaml") or {}).get("items", [])
    massign = result["res"].get("model_assign", result["res"]["assign"])
    def _model_split(label):
        return len({c for k, c in massign.items() if k.rsplit("|", 1)[0] == label.upper()}) > 1
    coll = bool(coll_ids) and all(_model_split(g["label"]) for g in gold_items if g["id"] in coll_ids)
    add(4, "alias resolution vs gold, MODEL ONLY: pairwise F1 >= 0.85, >= 80% of multi-label characters fully merged, every gold collision split",
        a["f1"] >= 0.85 and coll and a.get("characters_fully_merged", 0) >= 0.8 * max(1, a.get("characters_with_several_labels", 1)), f"P={a['pairwise_precision']} R={a['pairwise_recall']} F1={a['f1']}; characters fully merged {a.get('characters_fully_merged')}/{a.get('characters_with_several_labels')}; collision found={coll}; wrong merges={a['wrong_merges'][:3]}")
    s = ev["seeded"]
    add(5, "seeded-error recall >= 0.70 and <= 2 spurious new flags",
        (s["pipeline_recall"] or 0) >= 0.70 and len(spurious) <= 2,
        f"recall {s['pipeline_recall']} on {n} seeded errors (by validity: " +
        ", ".join(f"{k} {v['caught']}/{v['n']}" for k, v in sorted(s.get("by_validity", {}).items())) + "); per type " +
        ", ".join(f"{k} {v['pipeline']}/{v['n']}" for k, v in sorted(s["per_type"].items())) + f"; spurious {len(spurious)}")
    add(6, "real-pack ACTIONABLE precision (contradiction/review/collision flags, all judged) >= 0.70",
        real["n_unlabelled"] == 0 and (real["precision"] or 0) >= 0.70,
        f"actionable precision {real['precision']} over {real['n_flags_scored']} flags ({real['n_unlabelled']} unlabelled); alias-list flags {real['alias_flag_precision']} over {real['n_alias_flags']}; distinct gold errors found {real['distinct_gold_found']}/{real['n_gold']}")
    add(7, "pipeline beats code-only baseline on seeded errors",
        (s["pipeline_recall"] or 0) > (s["baseline_recall"] or 0),
        f"pipeline {s['pipeline_recall']} vs baseline {s['baseline_recall']}")
    add(8, "model-swap test (different provider adapter + model, held-out episodes)", *swap_test(data, out))
    # 9 determinism: re-run from cache into a temp folder and compare flags.json bytes
    det_out = out / "_determinism"
    r2 = run_pipeline(data, det_out, result["summary"]["show"], ROOT / "review" / "aliases.yaml", 3, log=lambda *a: None)
    same = sha(det_out / "flags.json") == sha(out / "flags.json")
    live = r2["runlog"]["usage"]["calls_live"]
    add(9, "re-run from cache is byte-identical (flags.json) with 0 live calls", same and live == 0,
        f"identical={same}, live calls on re-run={live}")
    shutil.rmtree(det_out, ignore_errors=True)
    # 12 human loop: dismiss one flag in a temp review folder, re-run from cache, confirm it is applied
    hl_dir = out / "_review_test"
    hl_dir.mkdir(parents=True, exist_ok=True)
    target = next((f for f in flags if f["type"] in NON_INFO), None)
    if target:
        (hl_dir / "aliases.yaml").write_text("aliases: []\nevents: []\n", encoding="utf-8")
        (hl_dir / "decisions.yaml").write_text(yaml.safe_dump({"decisions": {target["id"]: {"decision": "dismiss"}}}), encoding="utf-8")
        r3 = run_pipeline(data, hl_dir / "run", result["summary"]["show"], hl_dir / "aliases.yaml", 3, log=lambda *a: None)
        st = {f["id"]: f.get("status") for f in r3["flags"]}
        ok_hl = st.get(target["id"]) == "dismissed" and sum(1 for v in st.values() if v == "dismissed") == 1 \
            and set(st) == {f["id"] for f in flags} and r3["runlog"]["usage"]["calls_live"] == 0
        add(12, "human loop: an editor's dismiss decision is applied on the next run (stable flag ids, 0 live calls)",
            ok_hl, f"dismissed {target['id']} -> status {st.get(target['id'])}; other flags unchanged={set(st) == {f['id'] for f in flags}}")
    shutil.rmtree(hl_dir, ignore_errors=True)
    # 13 held-out fictional show (written in iter 7, never used for tuning): run and report as-is
    held_dir = ROOT / "demo" / "heldout"
    if (held_dir / "show").exists():
        saved = GOLD_DIR
        try:
            GOLD_DIR = held_dir / "gold"
            hres = run_pipeline(held_dir / "show", out / "heldout_run", "held-out show", None, 3, log=lambda *a: None)
            hreal = score_real(hres["flags"])
            hreal["loss_stage"] = loss_stages(hres)
            hal = score_aliases(hres["res"].get("model_assign", hres["res"]["assign"]))
            ev["heldout"] = {"real": hreal, "aliases": hal, "summary": hres["summary"],
                             "ci95": wilson(hreal.get("distinct_gold_found", 0), hreal.get("n_gold", 0))}
            add(13, "held-out fictional show runs end to end; recall reported as-is (not a tuning target)", True,
                f"found {hreal.get('distinct_gold_found')}/{hreal.get('n_gold')} planted errors "
                f"(95% CI {ev['heldout']['ci95']}), actionable precision {hreal.get('precision')}, alias F1 {hal['f1']}; "
                f"missed {[k for k, v in hreal['found'].items() if not v]}")
        except Exception as e:
            add(13, "held-out fictional show runs end to end", False, f"crashed: {e}")
        finally:
            GOLD_DIR = saved
    # 14 resilience: with the resolver and the judge simulated as unavailable, the run still finishes with a report
    demo = ROOT / "demo" / "show"
    if demo.exists():
        saved_fail = os.environ.get("LLM_FAIL_TAGS")
        os.environ["LLM_FAIL_TAGS"] = "resolve,adjudicate"
        try:
            rr = run_pipeline(demo, out / "_resilience", "resilience", None, 3, log=lambda *a: None)
            ok_r = (out / "_resilience" / "report.html").exists() and rr["summary"]["identities_code_only"] \
                and rr["summary"]["groups_not_judged"] == rr["summary"]["candidate_groups"]
            add(14, "resilience: resolver + judge unavailable -> run still finishes, everything sent to a human", ok_r,
                f"code-only identities={rr['summary']['identities_code_only']}, groups not judged {rr['summary']['groups_not_judged']}/{rr['summary']['candidate_groups']}, flags {len(rr['flags'])}")
        except Exception as e:
            add(14, "resilience: resolver + judge unavailable -> run still finishes", False, f"crashed: {e}")
        finally:
            if saved_fail is None:
                os.environ.pop("LLM_FAIL_TAGS", None)
            else:
                os.environ["LLM_FAIL_TAGS"] = saved_fail
            shutil.rmtree(out / "_resilience", ignore_errors=True)
    rl = json.loads((out / "run_log.json").read_text(encoding="utf-8"))
    add(10, "run log records calls, tokens, wall time, est. cost",
        all(k in rl for k in ("usage", "wall_seconds", "est_cost_usd_paid_tier")),
        f"{rl['usage']['calls_total']} calls, {rl['usage']['in_tokens']}+{rl['usage']['out_tokens']} tokens, {rl['wall_seconds']} s, ${rl['est_cost_usd_paid_tier']}")
    add(11, "real-pack recall vs the answer key (agent-drafted, every item cites its script line) >= 0.70", (real["recall"] or 0) >= 0.70,
        f"recall {real['recall']}: found {[k for k, v in real['found'].items() if v]}, missed {[k for k, v in real['found'].items() if not v]}")

    (out / "eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    passed = sum(c["pass"] for c in checks)
    print(f"\nDETERMINISTIC CHECKS: {passed}/{len(checks)} passed")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['n']:>2}. {c['check']}\n         {c['detail']}")
    return passed == len(checks)
