"""End-to-end ProRes probe, calibration still, and a preview-scale unwrap."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from geometry import LensModel, delivery_raster
from main import draw_calibration_overlay, main
from media_io import probe_media


def _encode_prores(
    path: Path,
    frames: list[np.ndarray],
    *,
    profile: int = 3,
    pix_fmt: str = "yuv422p10le",
    fps: int = 24,
    timecode: str = "01:00:00:00",
    audio: bool = False,
    comment: str | None = None,
) -> None:
    height, width = frames[0].shape[:2]
    raw = b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-color_range",
        "pc",
        "-i",
        "pipe:0",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-shortest"]
    cmd += [
        "-c:v",
        "prores_ks",
        "-profile:v",
        str(profile),
        "-pix_fmt",
        pix_fmt,
        "-vendor",
        "apl0",
        "-alpha_bits",
        "0",
        "-timecode",
        timecode,
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "arib-std-b67",
        "-colorspace",
        "bt2020nc",
    ]
    if audio:
        cmd += ["-c:a", "pcm_s16le"]
    if comment:
        cmd += ["-metadata", f"comment={comment}"]
    cmd.append(str(path))
    subprocess.run(cmd, input=raw, check=True)


def _marker_frame(lens: LensModel, width: int = 320, height: int = 180) -> np.ndarray:
    image = np.full((height, width, 3), 800, dtype=np.uint16)
    radius = float(lens.radius_at_theta(90.0))
    cx = lens.center_x
    cy = lens.center_y - radius
    yy, xx = np.ogrid[:height, :width]
    image[(xx - cx) ** 2 + (yy - cy) ** 2 <= 4.0**2] = 62000
    return image


@pytest.fixture(scope="module")
def marked_clip(tmp_path_factory) -> tuple[Path, LensModel]:
    folder = tmp_path_factory.mktemp("plates")
    lens = LensModel(center_x=160.0, center_y=90.0, radius=80.0)
    path = folder / "A001_C001.mov"
    frame = _marker_frame(lens)
    _encode_prores(path, [frame, frame, frame], audio=True, comment="hal250-reel")
    return path, lens


def _lens_args(lens: LensModel) -> list[str]:
    return [
        "--center-x",
        str(lens.center_x),
        "--center-y",
        str(lens.center_y),
        "--radius",
        str(lens.radius),
        "--no-unsharp",
    ]


def test_overlay_draws_horizon_and_edge():
    lens = LensModel(center_x=160.0, center_y=90.0, radius=80.0)
    preview = draw_calibration_overlay(np.zeros((180, 320, 3), dtype=np.uint16), lens)
    green = int(np.all(preview == (0, 220, 0), axis=2).sum())
    yellow = int(np.all(preview == (0, 220, 255), axis=2).sum())
    assert green > 80
    assert yellow > 80


def test_preview_render_preserves_prores_and_places_the_front(marked_clip, tmp_path):
    source, lens = marked_clip
    output = tmp_path / "A001_latlong.mov"
    code = main(
        [
            str(source),
            "-o",
            str(output),
            "--preview-scale",
            "16",
            "--overwrite",
            *_lens_args(lens),
        ]
    )
    assert code == 0, "render failed"
    raster = delivery_raster(125.0, preview_scale=16)
    info = probe_media(output)
    assert info.profile == 3
    assert info.codec_tag == "apch"
    assert info.pix_fmt == "yuv422p10le"
    assert info.width == raster.delivery_width
    assert info.height == raster.full_height
    assert info.frame_count == 3
    assert info.timecode == "01:00:00:00"
    assert info.color_primaries == "bt2020"
    assert info.color_transfer == "arib-std-b67"
    assert info.color_space == "bt2020nc"
    assert info.has_audio is True
    assert info.fps == probe_media(source).fps
    assert b"apl0" in output.read_bytes()
    tags = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(output)],
            text=True,
        )
    )["format"]["tags"]
    assert tags.get("comment") == "hal250-reel"

    decoded = _decode_rgb48(output)
    assert decoded.shape[0] == 3
    row, col = _bright_centroid(decoded[0])
    assert col == pytest.approx(raster.delivery_width / 2, abs=6)
    assert row == pytest.approx(0.72 * raster.active_height - 0.5, abs=6)
    # The uncaptured nadir pad is black, below the active strip.
    assert int(decoded[0, -1].max()) < 50
    assert int(decoded[0].max()) > 20000

    again = main([str(source), "-o", str(output), "--preview-scale", "16", *_lens_args(lens)])
    assert again == 1

    shifted = tmp_path / "shifted.mov"
    code = main(
        [
            str(source),
            "-o",
            str(shifted),
            "--preview-scale",
            "16",
            "--start-frame",
            "1",
            "--end-frame",
            "3",
            *_lens_args(lens),
        ]
    )
    assert code == 0
    shifted_info = probe_media(shifted)
    assert shifted_info.frame_count == 2
    assert shifted_info.timecode == "01:00:00:01"


def test_active_crop_4444_profile(tmp_path):
    source = tmp_path / "plate4444.mov"
    frame = np.full((64, 64, 3), 4000, dtype=np.uint16)
    _encode_prores(source, [frame, frame], profile=4, pix_fmt="yuv444p10le", audio=False)
    output = tmp_path / "active.mov"
    code = main(
        [
            str(source),
            "-o",
            str(output),
            "--output-mode",
            "active-crop",
            "--preview-scale",
            "32",
            "--center-x",
            "32",
            "--center-y",
            "32",
            "--radius",
            "30",
            "--no-unsharp",
        ]
    )
    assert code == 0
    info = probe_media(output)
    raster = delivery_raster(125.0, preview_scale=32)
    assert info.profile == 4
    assert info.codec_tag == "ap4h"
    assert info.pix_fmt in {"yuv444p10le", "yuv444p12le"}
    assert info.width == raster.delivery_width
    assert info.height == raster.active_height
    assert info.has_audio is False


def test_calibrate_writes_overlay_and_sidecar(marked_clip, tmp_path):
    source, lens = marked_clip
    png = tmp_path / "check.png"
    code = main(
        [
            str(source),
            "--calibrate",
            "--frame",
            "1",
            "--calibrate-output",
            str(png),
            *_lens_args(lens),
            "--k1",
            "0.1",
        ]
    )
    assert code == 0
    assert png.is_file()
    sidecar = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["frame"] == 1
    assert sidecar["center_x"] == pytest.approx(lens.center_x)
    assert sidecar["k1"] == pytest.approx(0.1)
    expected = LensModel(
        center_x=lens.center_x,
        center_y=lens.center_y,
        radius=lens.radius,
        k1=0.1,
    )
    assert sidecar["horizon_radius_px"] == pytest.approx(expected.radius_at_theta(90.0))
    image = pytest.importorskip("cv2").imread(str(png), flags=1)
    assert image is not None
    assert int(np.all(image == (0, 220, 0), axis=2).sum()) > 50


def test_rejects_non_prores_and_missing_ai_model(marked_clip, tmp_path):
    source, lens = marked_clip
    jpeg = tmp_path / "not_prores.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=24:d=1",
            "-frames:v",
            "2",
            "-c:v",
            "mjpeg",
            str(jpeg),
        ],
        check=True,
    )
    assert main([str(jpeg), "-o", str(tmp_path / "out.mov"), "--preview-scale", "32"]) == 1
    assert (
        main(
            [
                str(source),
                "--upscaler",
                "ai",
                "--dry-run",
                "--preview-scale",
                "32",
                *_lens_args(lens),
            ]
        )
        == 1
    )
    assert main([str(tmp_path / "missing.mov"), "-o", str(tmp_path / "no.mov")]) == 1


def test_batch_dry_run_writes_nothing(tmp_path):
    day = tmp_path / "day"
    day.mkdir()
    frame = np.full((48, 64, 3), 1000, dtype=np.uint16)
    _encode_prores(day / "A.mov", [frame])
    _encode_prores(day / "B.mov", [frame])
    out = tmp_path / "renders"
    code = main(
        [
            str(day),
            "--output-dir",
            str(out),
            "--dry-run",
            "--preview-scale",
            "32",
            "--center-x",
            "32",
            "--center-y",
            "24",
            "--radius",
            "20",
        ]
    )
    assert code == 0
    assert not out.exists() or not any(out.glob("*.mov"))


def test_gui_command_builder():
    from gui import build_command

    command = build_command(
        {
            "input": "/media/A001.mov",
            "output": "/renders/A001.mov",
            "output_dir": "",
            "output_mode": "active-crop",
            "upscaler": "lanczos",
            "interp": "cubic",
            "center_x": "3071.5",
            "center_y": "",
            "radius": "1616",
            "yaw_offset": "90",
            "theta_max": "",
            "preview_scale": "4",
            "sr_model": "",
            "device": "cpu",
            "calibration": "",
            "calibrate": "1",
            "overwrite": "1",
            "no_unsharp": "0",
        }
    )
    assert command[2] == "/media/A001.mov"
    assert "--output-mode" in command and "active-crop" in command
    assert "--calibrate" in command
    assert "--overwrite" in command
    assert "--center-y" not in command
    with pytest.raises(ValueError):
        build_command({"input": "  "})


def _bright_centroid(frame: np.ndarray) -> tuple[float, float]:
    peak = frame.max(axis=2).astype(np.float64)
    mask = peak >= max(1.0, float(peak.max()) * 0.5)
    ys, xs = np.nonzero(mask)
    weights = peak[ys, xs]
    return float(np.average(ys, weights=weights)), float(np.average(xs, weights=weights))


def _decode_rgb48(path: Path) -> np.ndarray:
    info = probe_media(path)
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
    )
    frame_bytes = info.width * info.height * 3 * 2
    count = len(raw) // frame_bytes
    return np.frombuffer(raw, dtype="<u2").reshape(count, info.height, info.width, 3)
