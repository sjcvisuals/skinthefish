"""Timecode math, ffprobe parsing, and ProRes dimension checks."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from media_io import (
    MediaError,
    check_prores_dimensions,
    delivery_notes,
    encoder_pix_fmt,
    encoder_plan,
    parse_probe_json,
    parse_timecode,
    shift_timecode,
)
from cuda_index import index_for_cuda
from resample import normalized_grid


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


def test_encoder_plan_keeps_16k_chunks_to_one_frame():
    workers, chunk = encoder_plan(15360, 8640, requested_workers=0)
    assert chunk == 1
    assert workers >= 1
    workers, chunk = encoder_plan(7680, 4320)
    assert chunk == 2
    preview_workers, preview_chunk = encoder_plan(960, 540)
    assert preview_chunk == 12
    assert encoder_plan(960, 540, requested_workers=2)[0] == 2


def test_cuda_wheel_follows_the_driver():
    assert index_for_cuda(12, 8) == "cu128"
    assert index_for_cuda(13, 0) == "cu128"
    assert index_for_cuda(12, 7) == "cu124"
    assert index_for_cuda(12, 4) == "cu124"


def test_gpu_grid_lands_on_the_same_pixel_center_as_opencv():
    # align_corners=False maps pixel i to (i + 0.5) / size * 2 - 1.
    # The inverse of normalized_grid must return the OpenCV map value.
    map_x = np.array([[0.0, 1.5, 10.25]], dtype=np.float32)
    map_y = np.array([[3.0, 4.0, 0.5]], dtype=np.float32)
    grid = normalized_grid(map_x, map_y, src_w=32, src_h=16)
    recovered_x = ((grid[..., 0] + 1.0) * 32 - 1.0) / 2.0
    recovered_y = ((grid[..., 1] + 1.0) * 16 - 1.0) / 2.0
    assert recovered_x == pytest.approx(map_x, abs=1e-5)
    assert recovered_y == pytest.approx(map_y, abs=1e-5)
