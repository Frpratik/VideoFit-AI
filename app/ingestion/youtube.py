"""YouTube URL validation, video ID extraction, and timestamp link building."""

import re
from urllib.parse import parse_qs, urlparse

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


class InvalidYouTubeURL(ValueError):
    """Raised when a string is not a usable YouTube video URL."""


def extract_video_id(url: str) -> str:
    """Return the 11-character video ID from a YouTube URL.

    Supports watch, youtu.be, shorts, embed and live URLs.
    Raises InvalidYouTubeURL for anything else.
    """
    url = (url or "").strip()
    if not url:
        raise InvalidYouTubeURL("Please enter a YouTube URL.")
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    candidate = None

    if host == "youtu.be" and parts:
        candidate = parts[0]
    elif host in YOUTUBE_HOSTS:
        if parts and parts[0] == "watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            candidate = parts[1]

    if not candidate or not VIDEO_ID_PATTERN.match(candidate):
        raise InvalidYouTubeURL("That does not look like a valid YouTube video URL.")
    return candidate


def timestamp_url(video_id: str, seconds: float) -> str:
    """Build a clickable link that opens the video at the given second."""
    if not VIDEO_ID_PATTERN.match(video_id):
        raise InvalidYouTubeURL(f"Invalid video ID: {video_id!r}")
    return f"https://www.youtube.com/watch?v={video_id}&t={max(0, int(seconds))}s"
