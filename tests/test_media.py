"""Timecode math, ffprobe parsing, and ProRes dimension checks."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from media_io import (
    MediaError,
    check_prores_dimensions,
    delivery_notes,
    encoder_pix_fmt,
    parse_probe_json,
    parse_timecode,
    shift_timecode,
)


def _video(**overrides) -> dict:
    stream = {
        "codec_type": "video",
        "codec_name": "prores",
        "codec_tag_string": "apch",
        "profile": "HQ",
        "width": 6144,
        "height": 3240,
        "avg_frame_rate": "24000/1001",
        "r_frame_rate": "24000/1001",
        "pix_fmt": "yuv422p10le",
        "nb_frames": "10",
        "color_primaries": "bt2020",
        "color_transfer": "arib-std-b67",
        "color_space": "bt2020nc",
        "color_range": "tv",
        "bits_per_raw_sample": "10",
        "tags": {"timecode": "01:00:00:00"},
    }
    stream.update(overrides)
    return stream


def test_profile_tag_wins_and_color_is_kept():
    info = parse_probe_json(
        {
            "streams": [
                _video(),
                {"codec_type": "audio", "codec_name": "pcm_s16le"},
                {"codec_type": "data", "codec_tag_string": "tmcd", "tags": {"timecode": "01:00:00:00"}},
            ],
            "format": {"duration": "0.4"},
        },
        Path("A001.mov"),
    )
    assert info.profile == 3
    assert info.profile_name == "hq"
    assert info.codec_tag == "apch"
    assert info.fps == Fraction(24000, 1001)
    assert info.frame_count == 10
    assert info.timecode == "01:00:00:00"
    assert info.drop_frame is False
    assert info.has_audio is True
    assert info.color_transfer == "arib-std-b67"
    assert info.color_range == "tv"
    assert encoder_pix_fmt(info.profile) == "yuv422p10le"
    assert delivery_notes(info) == []


def test_4444_tag_and_alpha_note():
    info = parse_probe_json(
        {"streams": [_video(codec_tag_string="ap4h", profile="4444", pix_fmt="yuva444p12le", bits_per_raw_sample="12")], "format": {}},
        Path("B002.mov"),
    )
    assert info.profile == 4
    assert info.has_alpha is True
    assert encoder_pix_fmt(4) == "yuv444p10le"
    assert encoder_pix_fmt(5) == "yuv444p10le"
    notes = " ".join(delivery_notes(info))
    assert "alpha" in notes
    assert "10-bit" in notes


def test_xq_tag():
    info = parse_probe_json(
        {"streams": [_video(codec_tag_string="ap4x", profile="XQ", pix_fmt="yuv444p12le")], "format": {}},
        Path("C.mov"),
    )
    assert info.profile == 5
    assert info.profile_name == "4444xq"


def test_non_prores_has_no_profile():
    info = parse_probe_json(
        {"streams": [_video(codec_name="h264", codec_tag_string="avc1", profile="High", pix_fmt="yuv420p")], "format": {}},
        Path("D.mp4"),
    )
    assert info.is_prores is False
    assert info.profile is None


def test_drop_frame_separator_is_preserved():
    info = parse_probe_json(
        {"streams": [_video(tags={"timecode": "00:01:00;02"}, avg_frame_rate="30000/1001")], "format": {}},
        Path("E.mov"),
    )
    assert info.drop_frame is True
    assert info.timecode == "00:01:00;02"


def test_non_drop_timecode_arithmetic():
    fps = Fraction(24, 1)
    assert shift_timecode("01:00:00:00", 0, fps, False) == "01:00:00:00"
    assert shift_timecode("01:00:00:00", 24, fps, False) == "01:00:01:00"
    assert shift_timecode("00:00:00:00", 86400, Fraction(24000, 1001), False) == "01:00:00:00"
    hours, minutes, seconds, frames, drop = parse_timecode("01:02:03:04")
    assert (hours, minutes, seconds, frames, drop) == (1, 2, 3, 4, False)


def test_drop_frame_timecode_skips_the_first_two_frame_numbers():
    fps = Fraction(30000, 1001)
    assert shift_timecode("00:00:00;00", 1800, fps, True) == "00:01:00;02"
    assert shift_timecode("00:00:59;29", 1, Fraction(30, 1), True) == "00:01:00;02"
    assert shift_timecode("00:01:00;02", 0, fps, True) == "00:01:00;02"
    # 60 fps drop-frame skips four frame numbers.
    assert shift_timecode("00:00:59;59", 1, Fraction(60000, 1001), True) == "00:01:00;04"


def test_prores_dimension_guard():
    check_prores_dimensions(15360, 7680)
    check_prores_dimensions(15360, 5333)
    with pytest.raises(MediaError):
        check_prores_dimensions(65536, 100)
    with pytest.raises(MediaError):
        check_prores_dimensions(15361, 7680)


def test_bad_timecode():
    with pytest.raises(MediaError):
        parse_timecode("1:00:00:00")
