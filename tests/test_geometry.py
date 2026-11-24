"""Optical model and delivery-raster locks for the HAL 250 unwrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry import (
    DEFAULT_RADIUS,
    HORIZON_THETA_DEG,
    NATIVE_WIDTH,
    UHD_HEIGHT,
    UHD_WIDTH,
    GeometryError,
    LensModel,
    autoscale_default_lens,
    build_remap_maps,
    delivery_raster,
    load_lens_json,
    save_lens_json,
)


def test_horizon_radius_is_72_percent_of_the_250_circle():
    lens = LensModel()
    horizon = lens.radius_at_theta(HORIZON_THETA_DEG)
    assert horizon == pytest.approx(1166.4, abs=1e-6)
    assert horizon / DEFAULT_RADIUS == pytest.approx(0.72, abs=1e-12)


def test_polynomial_keeps_the_calibrated_edge_and_bends_the_middle():
    bent = LensModel(k1=0.2)
    assert bent.radius_at_theta(125.0) == pytest.approx(DEFAULT_RADIUS)
    # t = 0.5 -> curve = 0.5 * (1 + 0.2 * 0.25) = 0.525, edge = 1.2
    assert bent.radius_at_theta(62.5) == pytest.approx(DEFAULT_RADIUS * 0.4375, abs=1e-6)
    assert LensModel().radius_at_theta(62.5) == pytest.approx(DEFAULT_RADIUS * 0.5)


def test_collapsed_polynomial_is_rejected():
    with pytest.raises(GeometryError):
        LensModel(k1=-1.0).radius_at_theta(10.0)


def test_delivery_raster_is_16_9_with_no_pad():
    raster = delivery_raster()
    assert (raster.native_width, raster.native_height) == (7850, 4416)
    assert (raster.delivery_width, raster.delivery_height) == (15360, 8640)
    assert raster.delivery_width * 9 == raster.delivery_height * 16
    assert raster.delivery_width == 4 * 3840
    assert raster.delivery_height == 4 * 2160
    assert NATIVE_WIDTH == 7850
    assert UHD_WIDTH == 15360
    assert UHD_HEIGHT == 8640
    assert raster.upscale_x == pytest.approx(15360 / 7850, rel=1e-12)
    assert raster.upscale_y == pytest.approx(8640 / 4416, rel=1e-12)


def test_preview_scale_keeps_exact_16_9():
    raster = delivery_raster(preview_scale=16)
    assert raster.delivery_width % 16 == 0
    assert raster.delivery_width * 9 == raster.delivery_height * 16
    assert raster.delivery_width < 15360


def test_front_direction_straddles_the_center_columns():
    lens = LensModel()
    width, height = 360, 125
    map_x, map_y = build_remap_maps(width, height, lens)
    # Height 125 puts pixel centers on 89.5° and 90.5°, so the horizon sits between them.
    left, right = width // 2 - 1, width // 2
    for row in (89, 90):
        theta = (row + 0.5) / height * 125.0
        radius = lens.radius_at_theta(theta)
        assert 0.5 * (map_x[row, left] + map_x[row, right]) == pytest.approx(lens.center_x, abs=0.05)
        assert map_y[row, left] == pytest.approx(lens.center_y - radius, abs=1.0)
        assert map_y[row, right] == pytest.approx(map_y[row, left], abs=0.05)


def test_positive_yaw_aims_the_center_column_to_image_right():
    lens = LensModel(yaw_offset_deg=90.0)
    map_x, map_y = build_remap_maps(360, 100, lens)
    row = 50
    theta = (row + 0.5) / 100 * 125.0
    radius = float(lens.radius_at_theta(theta))
    # The two central pixels sit ±0.5° off the exact center column.
    expected_x = lens.center_x + radius * math.cos(math.radians(0.5))
    xs = 0.5 * (map_x[row, 179] + map_x[row, 180])
    ys = 0.5 * (map_y[row, 179] + map_y[row, 180])
    assert xs == pytest.approx(expected_x, abs=0.2)
    assert ys == pytest.approx(lens.center_y, abs=0.2)


def test_seam_columns_sample_neighboring_azimuths():
    map_x, map_y = build_remap_maps(360, 100, LensModel())
    row = 71  # near the horizon
    seam = math.hypot(map_x[row, 0] - map_x[row, -1], map_y[row, 0] - map_y[row, -1])
    quarter = math.hypot(map_x[row, 0] - map_x[row, 90], map_y[row, 0] - map_y[row, 90])
    assert seam < quarter / 20


def test_master_grid_puts_the_horizon_on_the_spec_radius():
    lens = LensModel()
    map_x, map_y = build_remap_maps(7850, 4416, lens)
    assert map_x.shape == (4416, 7850)
    assert map_x.dtype == np.float32
    # θ = 90° falls between rows whose centers bracket 0.72 of the frame.
    target = 0.72 * 4416 - 0.5
    row = int(math.floor(target))
    theta = (row + 0.5) / 4416 * 125.0
    radius = float(lens.radius_at_theta(theta))
    col_a, col_b = 3924, 3925  # front straddles the middle of 7850
    mid_x = 0.5 * (float(map_x[row, col_a]) + float(map_x[row, col_b]))
    mid_y = 0.5 * (float(map_y[row, col_a]) + float(map_y[row, col_b]))
    assert mid_x == pytest.approx(3072.0, abs=0.05)
    assert mid_y == pytest.approx(1620.0 - radius, abs=0.05)
    assert radius == pytest.approx(1166.4, abs=0.5)


def test_proxy_frame_scales_the_default_lens_only():
    scaled = autoscale_default_lens(LensModel(), 3072, 1620)
    assert scaled.center_x == pytest.approx(1536.0)
    assert scaled.center_y == pytest.approx(810.0)
    assert scaled.radius == pytest.approx(810.0)
    weird = autoscale_default_lens(LensModel(), 1000, 1000)
    assert weird.radius == pytest.approx(DEFAULT_RADIUS)


def test_calibration_json_roundtrip(tmp_path):
    path = tmp_path / "lens.json"
    lens = LensModel(center_x=3071.5, center_y=1619.25, radius=1616.0, yaw_offset_deg=12.5, k1=-0.01)
    save_lens_json(path, lens, notes="stage")
    loaded = load_lens_json(path)
    assert loaded == lens


def test_bright_disk_at_the_front_horizon_lands_in_the_center():
    cv2 = pytest.importorskip("cv2")
    lens = LensModel(center_x=160.0, center_y=90.0, radius=80.0)
    image = np.zeros((180, 320, 3), dtype=np.uint16)
    radius = float(lens.radius_at_theta(90.0))
    cx = lens.center_x
    cy = lens.center_y - radius
    yy, xx = np.ogrid[:180, :320]
    image[(xx - cx) ** 2 + (yy - cy) ** 2 <= 2.5**2] = 50000
    map_x, map_y = build_remap_maps(360, 125, lens)
    unwrapped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    row, col = _bright_centroid(unwrapped)
    assert col == pytest.approx(179.5, abs=1.5)
    assert row == pytest.approx(89.5, abs=1.5)


def _bright_centroid(frame: np.ndarray) -> tuple[float, float]:
    peak = frame.max(axis=2).astype(np.float64)
    mask = peak >= max(1.0, peak.max() * 0.5)
    ys, xs = np.nonzero(mask)
    weights = peak[ys, xs]
    return float(np.average(ys, weights=weights)), float(np.average(xs, weights=weights))
