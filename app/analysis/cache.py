"""Remember topic assessments so re-running an analysis does not spend tokens twice.

A saved judgement is reused only when EVERYTHING that could change it is identical: the transcript text,
the video, the topic wording, the answer language, the model, and the assessment prompt. The cache is a small
JSON file next to the vector index, so it survives restarts.
"""

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional

from app.schemas import Segment, Topic, TopicResult

MAX_ENTRIES = 2000


def transcript_hash(segments: list[Segment]) -> str:
    """Fingerprint of the parsed transcript (ids, times and text)."""
    digest = hashlib.sha256()
    for s in segments:
        digest.update(f"{s.id}|{s.start_sec}|{s.text}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


class AssessmentCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {}  # unreadable cache: start empty, never crash

    @staticmethod
    def key(prompt_version: str, model: str, video_id: str, text_hash: str, language: str, topic: Topic) -> str:
        parts = [prompt_version, model, video_id, text_hash, language, topic.name.strip().lower(),
                 topic.description.strip().lower()]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str, topic: Topic) -> Optional[TopicResult]:
        with self._lock:
            raw = self._data.get(key)
        if raw is None:
            return None
        try:
            # importance only changes score weighting, so always use the caller's current topic
            return TopicResult(**raw).model_copy(update={"topic": topic, "from_cache": True})
        except ValueError:
            return None

    def put(self, key: str, result: TopicResult) -> None:
        if not result.assessed:
            return  # never remember a failure
        with self._lock:
            self._data[key] = result.model_copy(update={"from_cache": False}).model_dump()
            while len(self._data) > MAX_ENTRIES:
                self._data.pop(next(iter(self._data)))
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # a cache that cannot be written is just a cache miss next time

    def __len__(self) -> int:
        return len(self._data)
