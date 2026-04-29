"""Minimal loader for the LoCoMo benchmark dataset.

The locomo repo lives in a sibling folder by default. Override with
LOCOMO_DIR (path to the directory containing locomo10.json) if your
checkout lives elsewhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# A handful of QA entries in locomo10.json pack multiple evidence dia_ids
# into a single string (e.g. "D8:6; D9:17" or "D9:1 D4:4 D4:6"). Split on
# whitespace, semicolons, and commas to recover individual ids.
_EVIDENCE_SPLIT = re.compile(r"[\s;,]+")


_DEFAULT_REL_PATH = Path(__file__).resolve().parent.parent.parent / "locomo" / "data"


def default_locomo_dir() -> Path:
    env = os.environ.get("LOCOMO_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _DEFAULT_REL_PATH


def locomo_json_path(locomo_dir: Path | None = None) -> Path:
    base = locomo_dir or default_locomo_dir()
    path = base / "locomo10.json"
    if not path.exists():
        raise FileNotFoundError(
            f"locomo10.json not found at {path}. Either clone "
            "https://github.com/snap-research/locomo into a sibling "
            "folder, or set LOCOMO_DIR to point at the data directory."
        )
    return path


@dataclass(frozen=True)
class Turn:
    speaker: str
    dia_id: str
    text: str
    img_url: str | None = None
    blip_caption: str | None = None


@dataclass(frozen=True)
class Session:
    session_id: str  # e.g. "session_1"
    date_time: str   # raw string from the dataset
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class QA:
    question: str
    answer: str | None
    evidence: tuple[str, ...]
    category: int
    adversarial_answer: str | None = None  # only set on category-5 questions


@dataclass(frozen=True)
class Sample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]
    qa: tuple[QA, ...]

    def turn_by_dia_id(self, dia_id: str) -> Turn | None:
        for session in self.sessions:
            for turn in session.turns:
                if turn.dia_id == dia_id:
                    return turn
        return None


def _normalize_evidence(raw: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for entry in raw:
        for piece in _EVIDENCE_SPLIT.split(entry):
            piece = piece.strip()
            if piece and piece != "D":
                out.append(piece)
    return tuple(out)


def _build_session(conv: dict, session_id: str) -> Session:
    raw_turns = conv[session_id]
    turns = tuple(
        Turn(
            speaker=t["speaker"],
            dia_id=t["dia_id"],
            text=t["text"],
            img_url=t.get("img_url"),
            blip_caption=t.get("blip_caption"),
        )
        for t in raw_turns
    )
    return Session(
        session_id=session_id,
        date_time=conv.get(f"{session_id}_date_time", ""),
        turns=turns,
    )


def _session_sort_key(session_id: str) -> int:
    # session_1, session_2, ... session_32
    return int(session_id.split("_", 1)[1])


def _build_sample(raw: dict) -> Sample:
    conv = raw["conversation"]
    session_ids = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=_session_sort_key,
    )
    sessions = tuple(_build_session(conv, sid) for sid in session_ids)
    qa = tuple(
        QA(
            question=q["question"],
            answer=q.get("answer"),
            evidence=_normalize_evidence(q.get("evidence", []) or []),
            category=q["category"],
            adversarial_answer=q.get("adversarial_answer"),
        )
        for q in raw.get("qa", [])
    )
    return Sample(
        sample_id=raw["sample_id"],
        speaker_a=conv["speaker_a"],
        speaker_b=conv["speaker_b"],
        sessions=sessions,
        qa=qa,
    )


def load_samples(locomo_dir: Path | None = None) -> list[Sample]:
    path = locomo_json_path(locomo_dir)
    with open(path) as f:
        raw = json.load(f)
    return [_build_sample(r) for r in raw]


def iter_qa(samples: list[Sample] | None = None) -> Iterator[tuple[str, QA]]:
    """Yield (sample_id, qa) across all samples, preserving dataset order."""
    if samples is None:
        samples = load_samples()
    for sample in samples:
        for qa in sample.qa:
            yield sample.sample_id, qa
