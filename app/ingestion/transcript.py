"""Parse a pasted transcript (YouTube copy, SRT, VTT, or plain text) into Segments."""

import re
from typing import Optional

from app.schemas import ParseResult, Script, Segment, TranscriptFormat

MAX_CHARS = 2_000_000

# 1:02:03  |  12:34  |  00:00:01,500  |  00:01.250
TIMESTAMP = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
TIMESTAMP_RE = re.compile(TIMESTAMP)
# A line that STARTS with a timestamp, optionally wrapped in [ ] or ( ), then text.
LINE_START_RE = re.compile(rf"^\s*[\[\(]?{TIMESTAMP}[\]\)]?\s*[-–:]?\s*(.*)$")
# YouTube's accessibility lines: "5 seconds", "1 minute, 2 seconds"
ACCESSIBILITY_RE = re.compile(
    r"^\d+\s+(?:hours?|minutes?|seconds?)(?:,?\s*\d+\s+(?:hours?|minutes?|seconds?))*$", re.I
)
TAG_RE = re.compile(r"<[^>]+>")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


class TranscriptError(ValueError):
    """Raised when the transcript is empty or unusable."""


def parse_timestamp(h: Optional[str], m: str, s: str, ms: Optional[str]) -> float:
    """Convert regex groups of a timestamp into seconds."""
    seconds = int(h or 0) * 3600 + int(m) * 60 + int(s)
    if ms:
        seconds += int(ms.ljust(3, "0")) / 1000
    return float(seconds)


def format_timestamp(seconds: float) -> str:
    """Seconds -> '1:02:03' or '2:05' for display."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def detect_script(text: str) -> Script:
    """Detect the writing system. Hindi/Marathi share Devanagari; Hinglish is Latin.

    This is NOT language detection: Romanized Hindi looks like 'latin'.
    """
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    total = dev + lat
    if total == 0:
        return "unknown"
    if dev / total >= 0.8:
        return "devanagari"
    if lat / total >= 0.8:
        return "latin"
    return "mixed"


def _valid(h: Optional[str], m: str, s: str) -> bool:
    """'12:75' or '1:99:00' are not real times; treat them as ordinary text."""
    return int(s) < 60 and (h is None or int(m) < 60)


def _line_start(line: str):
    """Regex groups if the line starts with a VALID timestamp, else None."""
    match = LINE_START_RE.match(line)
    if match and _valid(*match.groups()[0:3]):
        return match.groups()
    return None


CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub("", CONTROL_RE.sub("", text))).strip()


def _wrap(sentence: str, limit: int = 300) -> list[str]:
    """Break an over-long run of text at word boundaries so every piece stays searchable."""
    pieces, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > limit:
            pieces.append(current)
            current = ""
        current = f"{current} {word}".strip()
    return pieces + [current] if current else pieces


def _make_segments(rows: list[tuple[Optional[float], Optional[float], str]]) -> list[Segment]:
    segments = []
    for start, end, text in rows:
        text = _clean(text)
        if text:
            segments.append(
                Segment(
                    id=f"seg_{len(segments) + 1:04d}",
                    start_sec=start,
                    end_sec=end,
                    text=text,
                    script=detect_script(text),
                )
            )
    return segments


def _detect_format(raw: str) -> TranscriptFormat:
    if raw.lstrip().upper().startswith("WEBVTT"):
        return "vtt"
    if "-->" in raw:
        return "srt"
    starts = sum(1 for line in raw.splitlines() if _line_start(line))
    return "timestamped_text" if starts >= 2 else "plain"


def _parse_cues(raw: str) -> list[tuple[Optional[float], Optional[float], str]]:
    """SRT and VTT share the 'start --> end' cue structure."""
    cue_re = re.compile(rf"{TIMESTAMP}\s*-->\s*{TIMESTAMP}")
    rows = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        lines = block.strip().split("\n")
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue  # WEBVTT header, NOTE, STYLE blocks
        match = cue_re.search(lines[idx])
        if not match:
            continue
        g = match.groups()
        if not (_valid(*g[0:3]) and _valid(*g[4:7])):
            continue  # impossible times such as 00:75:00
        start = parse_timestamp(*g[0:4])
        end = parse_timestamp(*g[4:8])
        rows.append((start, end, " ".join(lines[idx + 1:])))
    return rows


def _parse_timestamped_text(raw: str) -> list[tuple[Optional[float], Optional[float], str]]:
    """Handles '0:05 text' and the YouTube copy style where '0:05' is on its own line."""
    entries: list[list] = []  # [start, text]
    for line in raw.splitlines():
        line = line.strip()
        if not line or ACCESSIBILITY_RE.match(line):
            continue
        g = _line_start(line)
        if g:
            entries.append([parse_timestamp(*g[0:4]), g[4]])
        elif entries:
            entries[-1][1] += " " + line  # continuation of the previous segment
        else:
            entries.append([None, line])  # text before the first timestamp
    rows = []
    for i, (start, text) in enumerate(entries):
        nxt = entries[i + 1][0] if i + 1 < len(entries) else None
        rows.append((start, nxt, text))
    return rows


def _parse_plain(raw: str) -> list[tuple[Optional[float], Optional[float], str]]:
    """No timestamps: split into sentence-sized pieces so retrieval still works."""
    rows, buffer = [], ""
    for line in raw.splitlines():
        for sentence in (p for s in SENTENCE_SPLIT_RE.split(line.strip()) for p in _wrap(s)):
            buffer = f"{buffer} {sentence}".strip()
            if len(buffer) >= 300:
                rows.append((None, None, buffer))
                buffer = ""
    if buffer:
        rows.append((None, None, buffer))
    return rows


def parse_transcript(raw: str) -> ParseResult:
    """Main entry point. Returns segments plus warnings about any limitation found."""
    raw = (raw or "").replace("﻿", "")  # files saved with a BOM would hide the first timestamp
    if not raw.strip():
        raise TranscriptError("The transcript is empty. Please paste the transcript text.")
    if len(raw) > MAX_CHARS:
        raise TranscriptError("The transcript is too large (over 2,000,000 characters).")

    fmt = _detect_format(raw)
    if fmt in ("srt", "vtt"):
        rows = _parse_cues(raw)
    elif fmt == "timestamped_text":
        rows = _parse_timestamped_text(raw)
    else:
        rows = _parse_plain(raw)

    segments = _make_segments(rows)
    if not segments:
        raise TranscriptError("No readable transcript text was found in the input.")

    warnings: list[str] = []
    timed = [s for s in segments if s.start_sec is not None]
    has_timestamps = len(timed) > 0

    if not has_timestamps:
        warnings.append(
            "No timestamps were found. Answers can quote the transcript but cannot link to video times."
        )
    else:
        if len(timed) < len(segments):
            warnings.append(f"{len(segments) - len(timed)} segment(s) have no timestamp.")
        starts = [s.start_sec for s in timed]
        if any(b < a for a, b in zip(starts, starts[1:])):
            warnings.append("Timestamps are not in increasing order; some citations may point to the wrong time.")
        if len(timed) == len(segments) and len(segments) > 1 and len(set(starts)) == 1:
            warnings.append("All segments share one timestamp; the timestamps look invalid.")

    return ParseResult(segments=segments, format=fmt, has_timestamps=has_timestamps, warnings=warnings)
