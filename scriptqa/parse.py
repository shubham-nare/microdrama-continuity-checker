"""Parse Kuku-format screenplays into timed elements with stable IDs. Pure code, no model.

Tolerates the quirks seen in the pack: upper-case scene headers (ep6), (O.S.)/(V.O.) modifiers inside
or outside the emotion parentheses, elements listed out of order (ep12), elements running past
their scene end (ep22).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = re.compile(r"Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)")
LABEL = re.compile(r"^\*\*(.+?):\*\*\s*$")
SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
HDR = re.compile(r"^(scene heading|scene start|scene end)\s*:\s*(.*)$", re.I)
EP_NUM = re.compile(r"^EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
NON_SPEAKER = {"CAMERA", "ACTION", "SOUND FX", "SFX", "SOUND", "MUSIC", "TEXT ON SCREEN", "TITLE"}
MODIFIER = re.compile(r"\b(O\.S\.|V\.O\.|O\.C\.|CONT'D|CONTINUES|CONT)\b", re.I)


def ts_to_s(m: str, s: str) -> float:
    return round(int(m) * 60 + float(s), 3)


def fmt_ts(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Element:
    id: str
    episode: int
    scene: int
    kind: str            # CAMERA | ACTION | SOUND FX | DIALOGUE | other label
    label: str           # raw label as printed (speaker label for dialogue)
    speaker: str | None  # normalised speaker label (modifiers stripped) for DIALOGUE
    emotion: str | None
    start: float
    end: float
    text: str
    line: int            # 1-based line of the element's text in the source file


@dataclass
class Scene:
    number: int
    heading: str
    start: float | None
    end: float | None


@dataclass
class Episode:
    number: int
    file: str
    logline: str
    continuity_notes: str
    scenes: list[Scene] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)
    raw_timestamp_lines: int = 0

    @property
    def duration(self) -> float:
        return max((e.end for e in self.elements), default=0.0)

    def by_id(self) -> dict[str, Element]:
        return {e.id: e for e in self.elements}


def split_label(raw: str) -> tuple[str, str, str | None]:
    """'LADY MAYA (O.S. / Low, Threatening)' -> kind, speaker, emotion."""
    name = raw.strip()
    emotion = None
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if m:
        name, inner = m.group(1).strip(), m.group(2).strip()
        parts = [p.strip() for p in inner.split("/")]
        rest = [p for p in parts if not MODIFIER.fullmatch(p.strip(" ."))]
        rest = [MODIFIER.sub("", p).strip(" ,.") for p in rest]
        emotion = ", ".join(p for p in rest if p) or None
    name = MODIFIER.sub("", name).strip(" ()./")
    up = name.upper()
    if up in NON_SPEAKER:
        return up, None, None
    return "DIALOGUE", re.sub(r"\s+", " ", up), emotion


def parse_file(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8").splitlines()
    num, logline, notes = None, "", ""
    for ln in lines[:6]:
        if (m := EP_NUM.match(ln)):
            num = int(m.group(1))
        elif ln.upper().startswith("EPISODE LOGLINE:"):
            logline = ln.split(":", 1)[1].strip()
        elif ln.upper().startswith("CONTINUITY NOTES:"):
            notes = ln.split(":", 1)[1].strip()
    if num is None:
        m = re.search(r"episode-(\d+)", path.name)
        num = int(m.group(1)) if m else 0
    ep = Episode(number=num, file=path.name, logline=logline, continuity_notes=notes)
    ep.raw_timestamp_lines = sum(1 for ln in lines if TS.search(ln))

    scene_no, cur_scene = 0, None
    i = 0
    idx = 0
    while i < len(lines):
        ln = lines[i].strip()
        if (m := SCENE.match(ln)):
            scene_no = int(m.group(1))
            cur_scene = Scene(scene_no, "", None, None)
            ep.scenes.append(cur_scene)
        elif (m := HDR.match(ln)) and cur_scene is not None:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "scene heading":
                cur_scene.heading = val
            else:
                t = re.match(r"(\d+):(\d+(?:\.\d+)?)", val)
                if t:
                    setattr(cur_scene, "start" if key == "scene start" else "end", ts_to_s(*t.groups()))
        elif (m := LABEL.match(ln)):
            raw = m.group(1)
            # next non-empty line = timestamps, following non-empty lines until blank/next label = text
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = TS.search(lines[j]) if j < len(lines) else None
            if tm:
                k = j + 1
                text_lines = []
                while k < len(lines) and lines[k].strip() and not LABEL.match(lines[k].strip()) \
                        and not SCENE.match(lines[k].strip()) and not lines[k].startswith("---"):
                    text_lines.append(lines[k].strip())
                    k += 1
                kind, speaker, emotion = split_label(raw)
                idx += 1
                text = " ".join(text_lines).strip()
                if kind == "DIALOGUE":
                    text = text.strip()
                ep.elements.append(Element(
                    id=f"E{num:03d}-{idx:04d}", episode=num, scene=scene_no, kind=kind, label=raw.strip(),
                    speaker=speaker, emotion=emotion,
                    start=ts_to_s(tm.group(1), tm.group(2)), end=ts_to_s(tm.group(3), tm.group(4)),
                    text=text, line=j + 2,
                ))
                i = k - 1
        i += 1
    return ep


def parse_folder(folder: Path) -> list[Episode]:
    eps = [parse_file(p) for p in sorted(Path(folder).glob("*.txt"))]
    eps.sort(key=lambda e: e.number)
    return eps


def render_for_model(ep: Episode) -> str:
    """Compact, ID-tagged rendering the model reads. The model must cite these IDs."""
    out = [f"EPISODE {ep.number}", f"LOGLINE: {ep.logline}", f"CONTINUITY NOTES: {ep.continuity_notes}", ""]
    cur = None
    for e in ep.elements:
        if e.scene != cur:
            cur = e.scene
            sc = next((s for s in ep.scenes if s.number == cur), None)
            out.append(f"-- SCENE {cur}: {sc.heading if sc else ''}")
        who = f"{e.speaker}" + (f" ({e.emotion})" if e.emotion else "") if e.kind == "DIALOGUE" else e.kind
        out.append(f"[{e.id}] {fmt_ts(e.start)} {who}: {e.text}")
    return "\n".join(out)


def episode_to_dict(ep: Episode) -> dict:
    d = asdict(ep)
    d["duration"] = ep.duration
    return d
