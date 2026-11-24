"""Seam-aware resize and the tile path used by the AI upscaler."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from upscaler import AIUpscaler, LanczosUpscaler, UpscaleError, resize_wrap_horizontal


def _cosine(width: int, height: int) -> np.ndarray:
    x = np.arange(width, dtype=np.float32)
    wave = (np.cos(2.0 * np.pi * x / width) * 0.5 + 0.5) * 50000.0 + 4000.0
    frame = np.broadcast_to(wave, (height, width)).copy()
    return np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint16)


def _sample_error(image: np.ndarray, src_w: int, dst_w: int, columns: np.ndarray) -> float:
    """Mean absolute error against the cosine sampled at OpenCV's pixel centers."""
    xs = (columns.astype(np.float64) + 0.5) * src_w / dst_w - 0.5
    expected = (np.cos(2.0 * np.pi * xs / src_w) * 0.5 + 0.5) * 50000.0 + 4000.0
    actual = image[image.shape[0] // 2, columns, 0].astype(np.float64)
    return float(np.mean(np.abs(actual - expected)))


def test_wrapped_lanczos_matches_the_periodic_signal_at_the_seam():
    src_w, src_h = 180, 24
    dst_w, dst_h = 350, 40
    image = _cosine(src_w, src_h)
    wrapped = resize_wrap_horizontal(image, dst_w, dst_h, interpolation=cv2.INTER_LANCZOS4)
    border = np.concatenate([np.arange(4), np.arange(dst_w - 4, dst_w)])
    center = np.arange(dst_w // 2 - 2, dst_w // 2 + 2)
    border_error = _sample_error(wrapped, src_w, dst_w, border)
    center_error = _sample_error(wrapped, src_w, dst_w, center)
    assert wrapped.shape == (dst_h, dst_w, 3)
    assert wrapped.dtype == np.uint16
    assert border_error < 300
    assert border_error < center_error * 3 + 40


def test_lanczos_upscaler_hits_the_requested_size_and_preserves_level():
    image = np.full((20, 32, 3), 20000, dtype=np.uint16)
    out = LanczosUpscaler(unsharp=False).upscale(image, 64, 36)
    assert out.shape == (36, 64, 3)
    assert out.dtype == np.uint16
    assert abs(int(out[18, 32, 0]) - 20000) < 40


def test_unsharp_does_not_leave_the_legal_range():
    image = _cosine(64, 16)
    out = LanczosUpscaler(amount=0.8, sigma=0.7, unsharp=True).upscale(image, 96, 24)
    assert int(out.min()) >= 0
    assert int(out.max()) <= 65535


def test_ai_tiles_wrap_the_seam_and_fit_the_exact_size():
    def double(nchw: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(nchw, 2, axis=2), 2, axis=3)

    image = np.zeros((32, 48, 3), dtype=np.uint16)
    image[:, :4] = 62000
    image[:, -4:] = 62000
    upscaler = AIUpscaler(backend=double, scale=2, tile=16, tile_pad=4, wrap_pad=8, modulo=1)
    exact = upscaler.upscale(image, 96, 64)
    assert exact.shape == (64, 96, 3)
    assert int(exact[:, :4, 0].min()) > 60000
    assert int(exact[:, -4:, 0].min()) > 60000
    fitted = upscaler.upscale(image, 90, 60)
    assert fitted.shape == (60, 90, 3)
    assert int(fitted[:, 0, 0].max()) > 40000
    assert int(fitted[:, -1, 0].max()) > 40000


def test_missing_model_is_an_error(tmp_path):
    with pytest.raises(UpscaleError):
        AIUpscaler(tmp_path / "missing.onnx")
