"""Web demo: the same pipeline as `python run.py`, behind a page reviewers can use without installing anything.

    streamlit run app.py

The model key comes from Streamlit secrets (GEMINI_API_KEY) or .env. The page runs run.py in a subprocess on a
temporary copy of the input, then shows the HTML report, the flags table and downloads. Uploaded scripts are
deleted after the run and get their own throwaway model cache.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/shubham-nare/microdrama-continuity-checker"
SAMPLES = {
    "Lighthouse (3 episodes, fictional demo show)": ROOT / "demo" / "show",
    "Held-out show (5 episodes, fictional, never tuned on)": ROOT / "demo" / "heldout" / "show",
}
GATE_BASE = ROOT / "demo" / "heldout" / "show"
GATE_SAMPLES = {"episode 6 (has planted continuity errors)": ROOT / "demo" / "gate" / "episode-006.txt",
                "episode 7 (clean)": ROOT / "demo" / "gate" / "episode-007.txt"}
MAX_EPISODES = 12            # per upload: keeps one run inside a few minutes and the free quota
MAX_BYTES = 400_000
DAILY_EPISODE_CAP = 120      # live episodes per day across all visitors (free tier: ~500 requests/day)
QUOTA_FILE = Path(tempfile.gettempdir()) / "scriptqa_web_quota.json"
ENV_KEYS = ("GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK_MODEL", "LLM_BASE_URL", "LLM_API_KEY")

st.set_page_config(page_title="Microdrama continuity checker", page_icon="🎬", layout="wide")


# ---------- helpers ----------
def run_env(cache_dir: Path | None) -> dict:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        for k in ENV_KEYS:
            if k in st.secrets:
                env[k] = str(st.secrets[k])
    except Exception:  # no secrets file locally: run.py reads .env itself
        pass
    if cache_dir:
        env["LLM_CACHE_DIR"] = str(cache_dir)
    return env


def quota_used() -> int:
    try:
        q = json.loads(QUOTA_FILE.read_text())
        return q.get(dt.date.today().isoformat(), 0)
    except Exception:
        return 0


def quota_add(n: int) -> None:
    today = dt.date.today().isoformat()
    try:
        QUOTA_FILE.write_text(json.dumps({today: quota_used() + n}))
    except Exception:
        pass


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", Path(name).name)[:80] or "episode.txt"


def split_pasted(text: str) -> list[str]:
    parts = re.split(r"(?m)^(?=\s*EPISODE NUMBER:)", text)
    return [p.strip() + "\n" for p in parts if p.strip()]


def run_pipeline(args: list[str], cache_dir: Path | None) -> tuple[int, str]:
    box = st.empty()
    lines: list[str] = []
    p = subprocess.Popen([sys.executable, str(ROOT / "run.py"), *args], cwd=ROOT, env=run_env(cache_dir),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:  # live log while the model reads the episodes
        lines.append(line.rstrip())
        box.code("\n".join(lines[-18:]), language=None)
    rc = p.wait(timeout=1200)
    box.empty()
    return rc, "\n".join(lines)


def collect(out: Path) -> dict:
    files = {}
    for name in ("report.html", "flags.csv", "flags.json", "summary.json", "bible.json", "reference_cards.json",
                 "aliases.json", "gate.json", "run_log.json", "code_only.json"):
        if (out / name).exists():
            files[name] = (out / name).read_bytes()
    fx = out / "fix"
    for name in ("fix_and_recheck.json", "fix.diff"):
        if (fx / name).exists():
            files[f"fix/{name}"] = (fx / name).read_bytes()
    return files


def check(kind: str, data: Path, show: str, cache_dir: Path | None, gate_file: Path | None = None,
          code_only: bool = False) -> None:
    work = Path(tempfile.mkdtemp(prefix="scriptqa_web_"))
    try:
        out = work / "out"
        args = ["--data", str(data), "--out", str(out), "--show", show, "--review", str(work / "review" / "aliases.yaml")]
        if gate_file:
            args += ["--gate", str(gate_file)]
        if code_only:
            args.append("--code-only")
        with st.spinner("Running… a fresh show takes about 20–40 s per episode on the free model; cached demos are instant."):
            rc, log = run_pipeline(args, cache_dir)
        st.session_state["result"] = {"kind": kind, "show": show, "rc": rc, "log": log, "files": collect(out),
                                      "gate": bool(gate_file), "code_only": code_only}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------- page ----------
st.title("🎬 Continuity checker for AI-written microdrama scripts")
st.markdown(
    "Finds the places where a series contradicts itself before it goes to video: a name spelled several ways, "
    "eyes or a scar that change, one speaker label used for two people, one person under several labels. "
    "**The model reads, code decides, a human confirms:** the model pulls facts from each episode with a verbatim quote, "
    "code checks every quote and builds the candidate problems, the model can only dismiss a candidate with evidence, "
    f"and an editor accepts or dismisses each flag. Source and full write-up: [GitHub]({REPO})."
)
st.caption("Kuku's own scripts are confidential and are not on this site. The sample shows here are fictional, written for this tool.")

tab_try, tab_gate, tab_how = st.tabs(["Check a show", "Check one new episode (gate)", "Script format & notes"])

with tab_try:
    src = st.radio("Input", ["Sample show", "Upload episode .txt files", "Paste episodes"], horizontal=True)
    code_only = st.toggle("Code only (no model: spelling variants and speaker labels, in seconds)", value=False)
    if src == "Sample show":
        name = st.selectbox("Sample", list(SAMPLES))
        with st.expander("Read the sample scripts"):
            for f in sorted(SAMPLES[name].glob("*.txt")):
                st.text_area(f.name, f.read_text(encoding="utf-8"), height=200, key=f"src-{name}-{f.name}")
        if st.button("Run the check", type="primary", key="run-sample"):
            check("sample", SAMPLES[name], name.split(" (")[0], None, code_only=code_only)
    else:
        texts: list[tuple[str, str]] = []
        if src == "Upload episode .txt files":
            ups = st.file_uploader("One .txt file per episode, in the format shown in 'Script format'", type="txt",
                                   accept_multiple_files=True)
            texts = [(safe_name(u.name), u.getvalue().decode("utf-8", errors="replace")) for u in ups or []]
        else:
            pasted = st.text_area("Paste one or more episodes; each starts with a line 'EPISODE NUMBER: Episode N'", height=260)
            texts = [(f"episode-{i + 1:03d}.txt", t) for i, t in enumerate(split_pasted(pasted))] if pasted.strip() else []
        st.warning("Uploaded text is sent to Google's Gemini free tier, which may use it to improve Google's models. "
                   "Do not upload confidential scripts.", icon="⚠️")
        total = sum(len(t.encode()) for _, t in texts)
        left = DAILY_EPISODE_CAP - quota_used()
        if texts:
            st.caption(f"{len(texts)} episode(s), {total // 1000} KB. Today's shared allowance on this demo: {max(left, 0)} episode(s).")
        if st.button("Run the check", type="primary", disabled=not texts, key="run-upload"):
            if len(texts) > MAX_EPISODES or total > MAX_BYTES:
                st.error(f"This demo takes up to {MAX_EPISODES} episodes and {MAX_BYTES // 1000} KB per run. "
                         "For a full show, run it locally: see the README.")
            elif not code_only and len(texts) > left:
                st.error("Today's free-model allowance for this demo is used up. Try 'Code only', a sample show, or run it locally.")
            else:
                tmp = Path(tempfile.mkdtemp(prefix="scriptqa_in_"))
                try:
                    data = tmp / "show"
                    data.mkdir()
                    for n, t in texts:
                        (data / n).write_text(t, encoding="utf-8")
                    if not code_only:
                        quota_add(len(texts))
                    check("upload", data, "Your show", tmp / "cache", code_only=code_only)  # throwaway cache
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

with tab_gate:
    st.markdown("Checks **one new episode** against everything already written (here: the 5-episode held-out show), "
                "the way a writers' room would before sending it to video. Exit 'blocked' means it contradicts earlier episodes.")
    g = st.radio("New episode", list(GATE_SAMPLES) + ["Upload one episode"], horizontal=True)
    up = st.file_uploader("Episode .txt", type="txt", key="gate-up") if g == "Upload one episode" else None
    if g != "Upload one episode":
        with st.expander("Read it"):
            st.text(GATE_SAMPLES[g].read_text(encoding="utf-8"))
    if st.button("Check this episode", type="primary", disabled=(g == "Upload one episode" and not up), key="run-gate"):
        if up is not None:
            if quota_used() + 1 > DAILY_EPISODE_CAP:
                st.error("Today's free-model allowance for this demo is used up.")
            else:
                tmp = Path(tempfile.mkdtemp(prefix="scriptqa_gate_"))
                try:
                    f = tmp / safe_name(up.name)
                    f.write_bytes(up.getvalue())
                    quota_add(1)
                    check("gate", GATE_BASE, "Held-out show + your episode", tmp / "cache", gate_file=f)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
        else:
            check("gate", GATE_BASE, "Held-out show", None, gate_file=GATE_SAMPLES[g])

with tab_how:
    st.markdown(f"""
**Format.** One plain-text file per episode, the format of Kuku's screenplay pack: a header (`EPISODE NUMBER:`,
`EPISODE LOGLINE:`), then scenes, each with timed blocks such as `**CAMERA:**`, `**ACTION:**` or
`**NAME (emotion):**`, each followed by `Start: (MM:SS.ss) | End: (MM:SS.ss)` and the text. Download a sample below.

**What you get.** Every flag cites at least two script lines with a verbatim quote (code checks each quote exists),
a confidence, and a suggested fix. For spelling flags it also shows which lines would change and re-runs the
spelling check on a patched copy (applied only once an editor accepts the flag).

**What it does not do.** Plot logic, timeline maths, checking generated video. On Kuku's real 50-episode pack it found
6 of 7 known errors; about half of all its flags were real (all 4 high-confidence ones were). Numbers and limits are in the
[write-up and README]({REPO}).

**Model.** Gemini free tier (`gemini-3.5-flash-lite`, fallback `gemini-3.1-flash-lite`), swappable through config
(`LLM_PROVIDER`, `LLM_MODEL`); an OpenAI-compatible adapter is included.
""")
    sample = (ROOT / "demo" / "show" / "episode-001.txt").read_bytes()
    st.download_button("Download a sample episode", sample, "episode-001.txt", "text/plain")

# ---------- results ----------
r = st.session_state.get("result")
if r:
    st.divider()
    files = r["files"]
    if r["code_only"] and "code_only.json" in files:
        co = json.loads(files["code_only.json"])
        st.subheader(f"Code-only result: {co['episodes']} episodes, {co['elements']} lines, no model calls")
        if co["spelling_groups"]:
            for grp in co["spelling_groups"]:
                st.write("Spelled several ways: " + ", ".join(f"**{m['name']}** ×{m['count']} (ep {', '.join(map(str, m['episodes']))})" for m in grp))
        else:
            st.write("No name spelled several ways.")
        st.write("Speaker labels and the episodes they speak in:")
        st.dataframe(pd.DataFrame([{"label": k, "episodes": ", ".join(map(str, v))} for k, v in co["speaker_labels"].items()]),
                     hide_index=True, width="stretch")
    elif "report.html" in files:
        if r["gate"] and "gate.json" in files:
            gj = json.loads(files["gate.json"])
            verdict = str(gj.get("verdict", ""))
            if gj.get("blocked"):
                st.error(f"{verdict}: episode {gj.get('episode')} contradicts the {gj.get('earlier_episodes')} earlier episodes.")
                for b in gj["blocked"]:
                    st.write("• " + b)
            elif r["rc"] == 1:
                st.warning(f"{verdict}: earlier episodes already disagree on a value this episode repeats; an editor must pick one.")
            else:
                st.success(f"{verdict or 'PASS'}: no continuity conflict involving episode {gj.get('episode')}.")
        if "summary.json" in files:
            s = json.loads(files["summary.json"])
            c = st.columns(5)
            c[0].metric("Episodes", s.get("episodes"))
            c[1].metric("Flags", s.get("flags_total"))
            c[2].metric("Contradictions", s.get("flags_contradiction"))
            c[3].metric("Label used for 2 people", s.get("flags_collision"))
            c[4].metric("Person under several labels", s.get("flags_alias"))
        if "flags.csv" in files:
            df = pd.read_csv(io.BytesIO(files["flags.csv"]), encoding="utf-8-sig")
            keep = [k for k in ("rank", "severity", "type", "summary", "episodes", "suggested_fix") if k in df.columns]
            if len(df):
                st.dataframe(df[keep], hide_index=True, width="stretch")
        st.subheader("Full report (each flag with its script lines)")
        # the report HTML-escapes all script text; scripts are also blocked inside the frame as a second guard
        csp = "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        st.iframe(csp + files["report.html"].decode("utf-8"), height=900)
    else:
        st.error("The run did not produce a report. Log below.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
        z.writestr("run_log.txt", r["log"])
    d = st.columns(3)
    d[0].download_button("Download all outputs (.zip)", buf.getvalue(), "continuity_check.zip", "application/zip")
    if "report.html" in files:
        d[1].download_button("Download report.html", files["report.html"], "report.html", "text/html")
    if "flags.csv" in files:
        d[2].download_button("Download flags.csv", files["flags.csv"], "flags.csv", "text/csv")
    with st.expander("Run log"):
        st.code(r["log"] or "(empty)", language=None)
