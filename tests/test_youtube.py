import pytest

from app.ingestion.youtube import InvalidYouTubeURL, extract_video_id, timestamp_url

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/watch?v={VID}&t=30s&list=PL123",
        f"youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"  https://m.youtube.com/watch?v={VID}  ",
    ],
)
def test_extract_video_id_valid(url):
    assert extract_video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    ["", "   ", "hello", "https://example.com/watch?v=" + VID,
     "https://www.youtube.com/watch?v=short",
     "https://www.youtube.com/", "https://youtube.com.evil.com/watch?v=" + VID],
)
def test_extract_video_id_invalid(url):
    with pytest.raises(InvalidYouTubeURL):
        extract_video_id(url)


def test_timestamp_url():
    assert timestamp_url(VID, 125.9) == f"https://www.youtube.com/watch?v={VID}&t=125s"
    assert timestamp_url(VID, -5).endswith("&t=0s")


def test_timestamp_url_rejects_bad_id():
    with pytest.raises(InvalidYouTubeURL):
        timestamp_url("bad", 10)
