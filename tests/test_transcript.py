import pytest

from app.ingestion.transcript import (
    TranscriptError,
    detect_script,
    format_timestamp,
    parse_transcript,
)

YT_PASTE_SEPARATE = """0:00
welcome to this course
0:05
5 seconds
today we learn gradient descent
1:02:03
1 hour, 2 minutes, 3 seconds
final words
"""

YT_PASTE_SAME_LINE = "0:00 welcome\n0:07 to the course\n0:15 let's begin"

SRT = """1
00:00:01,000 --> 00:00:04,500
नमस्ते दोस्तों

2
00:00:04,500 --> 00:00:09,000
आज हम CNN सीखेंगे
"""

VTT = """WEBVTT

NOTE a comment

intro
00:01.000 --> 00:04.000
<c>Hello</c> everyone

00:04.000 --> 00:08.250
this is <b>backprop</b>
"""


def test_youtube_paste_timestamp_on_own_line():
    r = parse_transcript(YT_PASTE_SEPARATE)
    assert r.format == "timestamped_text"
    assert [s.start_sec for s in r.segments] == [0, 5, 3723]
    assert r.segments[1].text == "today we learn gradient descent"  # "5 seconds" removed
    assert r.segments[0].end_sec == 5
    assert r.segments[2].end_sec is None
    assert r.has_timestamps and not r.warnings


def test_timestamp_and_text_on_same_line():
    r = parse_transcript(YT_PASTE_SAME_LINE)
    assert [s.start_sec for s in r.segments] == [0, 7, 15]
    assert r.segments[2].text == "let's begin"


def test_srt_hindi_preserved():
    r = parse_transcript(SRT)
    assert r.format == "srt"
    assert r.segments[0].start_sec == 1.0 and r.segments[0].end_sec == 4.5
    assert r.segments[0].text == "नमस्ते दोस्तों"
    assert r.segments[0].script == "devanagari"
    assert r.segments[1].script == "mixed"


def test_vtt_short_timestamps_and_tags_removed():
    r = parse_transcript(VTT)
    assert r.format == "vtt"
    assert len(r.segments) == 2
    assert r.segments[0].text == "Hello everyone"
    assert r.segments[1].start_sec == 4.0 and r.segments[1].end_sec == 8.25


def test_plain_text_without_timestamps_warns():
    r = parse_transcript("This is a transcript. It has no times. Second sentence here.")
    assert r.format == "plain"
    assert not r.has_timestamps
    assert r.segments[0].start_sec is None
    assert any("No timestamps" in w for w in r.warnings)


def test_byte_order_mark_is_ignored():
    r = parse_transcript("﻿0:00\nhello\n0:05\nworld")
    assert r.segments[0].start_sec == 0 and not r.warnings


def test_decreasing_timestamps_warn():
    r = parse_transcript("0:30 a\n0:10 b\n0:20 c")
    assert any("not in increasing order" in w for w in r.warnings)


def test_segment_ids_unique_and_ordered():
    r = parse_transcript(YT_PASTE_SAME_LINE)
    assert [s.id for s in r.segments] == ["seg_0001", "seg_0002", "seg_0003"]


@pytest.mark.parametrize("raw", ["", "   \n\n", None])
def test_empty_raises(raw):
    with pytest.raises(TranscriptError):
        parse_transcript(raw)


def test_malformed_cue_only_raises():
    with pytest.raises(TranscriptError):
        parse_transcript("WEBVTT\n\n00:01.000 --> 00:02.000\n\n")


def test_too_large_raises():
    with pytest.raises(TranscriptError):
        parse_transcript("a" * 2_000_001)


def test_format_timestamp():
    assert format_timestamp(125) == "2:05"
    assert format_timestamp(3723) == "1:02:03"
    assert format_timestamp(-3) == "0:00"


@pytest.mark.parametrize(
    "text,expected",
    [("hello world", "latin"), ("नमस्ते दोस्तों", "devanagari"),
     ("aaj hum CNN seekhenge", "latin"), ("आज हम CNN सीखेंगे", "mixed"), ("123 ...", "unknown")],
)
def test_detect_script(text, expected):
    assert detect_script(text) == expected
