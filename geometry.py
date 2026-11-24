"""Polar mapping from an Entaniya HAL 250 zenith fisheye to equirectangular.

The lens is equidistant, ``r = f * θ``, and is mounted on a RED KOMODO 6K
pointing straight up. The optical axis is the zenith (polar angle 0). The
recorded circle covers 250° of field of view, which is polar angle 0° at the
center through 125° at the circle edge (35° below the horizon).

The delivery raster is 16:9 and filled by the captured polar range. There is
no black pad. ``16k`` is 15360×8640 (four times UHD). ``8k`` is 7680×4320
(two times UHD). Both use the same Lanczos scale, about 195.7%.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger("hal250.geometry")

# RED KOMODO 6K active sensor.
SENSOR_WIDTH = 6144
SENSOR_HEIGHT = 3240
SENSOR_WIDTH_MM = 27.03
SENSOR_HEIGHT_MM = 14.26
PIXEL_PITCH_UM = 4.4

# Entaniya HAL 250 3.6 mm. The 250° image circle is the full sensor height
# when the lens is centered, and the 180° horizon sits at 72% of that radius.
DEFAULT_CENTER_X = 3072.0
DEFAULT_CENTER_Y = 1620.0
DEFAULT_RADIUS = 1620.0
DEFAULT_THETA_MAX_DEG = 125.0
HORIZON_THETA_DEG = 90.0
HORIZON_FRACTION = HORIZON_THETA_DEG / DEFAULT_THETA_MAX_DEG  # 0.72

# 16:9 delivery. Both sides divide by 16, so ProRes macroblocks line up
# and nothing is padded. 16K is four times UHD; 8K is two times UHD.
UHD_WIDTH = 15360
UHD_HEIGHT = 8640
EIGHTK_WIDTH = 7680
EIGHTK_HEIGHT = 4320
UHD_ASPECT_W = 16
UHD_ASPECT_H = 9
MAX_POLAR_ANGLE_DEG = 180.0
DELIVERY_SIZES = ("16k", "8k")

# Working resolution before the ~195.7% Lanczos resize.
# 8K is half of 16K on both the unwrap and the delivery, so the scale matches.
# Heights are the 16:9 partners of those widths (round(7850 × 9/16) = 4416).
NATIVE_WIDTH = 7850
NATIVE_HEIGHT = 4416
_RASTERS = {
    "16k": (UHD_WIDTH, UHD_HEIGHT, NATIVE_WIDTH, NATIVE_HEIGHT),
    "8k": (EIGHTK_WIDTH, EIGHTK_HEIGHT, NATIVE_WIDTH // 2, NATIVE_HEIGHT // 2),
}

_CALIBRATION_FIELDS = (
    "center_x",
    "center_y",
    "radius",
    "theta_max_deg",
    "k1",
    "k2",
    "k3",
    "yaw_offset_deg",
)


class GeometryError(ValueError):
    """The lens model or the requested raster cannot be built."""


def round_half_up(value: float) -> int:
    """Round half away from zero for non-negative delivery sizes."""
    return int(math.floor(value + 0.5))


@dataclass(frozen=True)
class LensModel:
    """Equidistant fisheye with an optional even polynomial on the angle.

    The base curve is ``r(θ) = R * (θ / θ_max)``. Coefficients ``k1``, ``k2``,
    and ``k3`` bend that curve and are renormalized so ``θ_max`` still lands
    on ``radius``. With the coefficients at zero this is the pure equidistant
    model ``r(θ) = R_250 * (θ / 125°)``.

    ``yaw_offset_deg`` rotates azimuth. Zero yaw puts sensor-up (negative
    image Y from the optical center) at the middle column of the panorama.
    A positive yaw rotates the sampled direction clockwise in the image, which
    moves panorama content to the left.
    """

    center_x: float = DEFAULT_CENTER_X
    center_y: float = DEFAULT_CENTER_Y
    radius: float = DEFAULT_RADIUS
    theta_max_deg: float = DEFAULT_THETA_MAX_DEG
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    yaw_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise GeometryError(f"radius must be positive, got {self.radius}")
        if not 0 < self.theta_max_deg <= MAX_POLAR_ANGLE_DEG:
            raise GeometryError(
                "theta_max_deg must be in (0, 180], "
                f"got {self.theta_max_deg}"
            )

    def scaled(self, factor: float) -> LensModel:
        """Scale center and radius for a uniformly resized proxy frame."""
        if factor <= 0:
            raise GeometryError(f"scale factor must be positive, got {factor}")
        return replace(
            self,
            center_x=self.center_x * factor,
            center_y=self.center_y * factor,
            radius=self.radius * factor,
        )

    def radius_at_theta(self, theta_deg: float | np.ndarray) -> float | np.ndarray:
        """Pixel radius of a polar angle, in source pixels."""
        theta = np.asarray(theta_deg, dtype=np.float64)
        t = theta / self.theta_max_deg
        curve = t * (1.0 + self.k1 * t**2 + self.k2 * t**4 + self.k3 * t**6)
        edge = 1.0 + self.k1 + self.k2 + self.k3
        if edge <= 0:
            raise GeometryError(
                "distortion coefficients collapse the radius at theta_max "
                f"(1 + k1 + k2 + k3 = {edge})"
            )
        radial = self.radius * (curve / edge)
        radial = np.maximum(radial, 0.0)
        if np.isscalar(theta_deg):
            return float(radial)
        return radial

    def to_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in _CALIBRATION_FIELDS}


@dataclass(frozen=True)
class RasterSpec:
    """Native unwrap size and the 16:9 delivery size derived from it."""

    native_width: int
    native_height: int
    delivery_width: int
    delivery_height: int
    preview_scale: int
    theta_max_deg: float
    size: str

    @property
    def upscale_x(self) -> float:
        return self.delivery_width / self.native_width

    @property
    def upscale_y(self) -> float:
        return self.delivery_height / self.native_height


def delivery_raster(
    theta_max_deg: float = DEFAULT_THETA_MAX_DEG,
    preview_scale: int = 1,
    size: str = "16k",
) -> RasterSpec:
    """Build the unwrap and delivery sizes.

    The frame is always 16:9. ``theta_max_deg`` does not change the pixel
    size; it is the polar angle that the full frame height represents.
    ``size`` selects the 15360×8640 master or the 7680×4320 master. Preview
    scales shrink that master and snap the width to a multiple of 16 so the
    16:9 height stays exact.
    """
    if size not in _RASTERS:
        raise GeometryError(f"size must be one of {DELIVERY_SIZES}, got {size!r}")
    if preview_scale < 1:
        raise GeometryError(f"preview_scale must be >= 1, got {preview_scale}")
    if not 0 < theta_max_deg <= MAX_POLAR_ANGLE_DEG:
        raise GeometryError(
            f"theta_max_deg must be in (0, 180], got {theta_max_deg}"
        )
    master_w, master_h, native_master_w, native_master_h = _RASTERS[size]
    if preview_scale == 1:
        delivery_w, delivery_h = master_w, master_h
        native_w, native_h = native_master_w, native_master_h
    else:
        delivery_w = max(16, int(round(master_w / preview_scale / 16.0)) * 16)
        delivery_h = delivery_w * UHD_ASPECT_H // UHD_ASPECT_W
        native_w = max(16, round_half_up(native_master_w / preview_scale))
        native_h = max(8, round_half_up(delivery_h * native_w / delivery_w))
    if delivery_w * UHD_ASPECT_H != delivery_h * UHD_ASPECT_W:
        raise GeometryError(
            f"delivery {delivery_w}×{delivery_h} is not {UHD_ASPECT_W}:{UHD_ASPECT_H}"
        )
    if native_w < 16 or native_h < 8 or delivery_w < 16 or delivery_h < 8:
        raise GeometryError(
            f"preview-scale {preview_scale} collapses the raster below a usable size"
        )
    return RasterSpec(
        native_width=native_w,
        native_height=native_h,
        delivery_width=delivery_w,
        delivery_height=delivery_h,
        preview_scale=preview_scale,
        theta_max_deg=theta_max_deg,
        size=size,
    )


def build_remap_maps(
    out_width: int,
    out_height: int,
    lens: LensModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute float32 ``(map_x, map_y)`` for :func:`cv2.remap`.

    Pixel centers are the sample locations. Column ``u`` reads azimuth

        φ = 2π (u + 0.5) / W - π + yaw

    so φ runs over ``[-π, π)`` and the front direction (φ = 0 at yaw 0)
    straddles the middle of an even-width frame. Row ``v`` reads polar angle

        θ = θ_max (v + 0.5) / H

    with the zenith at the top and ``θ_max`` at the bottom of the frame.
    The horizon (90°) is 72% of the way down when ``θ_max`` is 125°.

    Image X grows right and Y grows down. At yaw 0, φ = 0 is toward the top
    of the sensor::

        x = cx + r(θ) sin φ
        y = cy - r(θ) cos φ
    """
    if out_width < 2 or out_height < 2:
        raise GeometryError(
            f"remap size must be at least 2×2, got {out_width}×{out_height}"
        )
    u = np.arange(out_width, dtype=np.float64)
    v = np.arange(out_height, dtype=np.float64)
    phi = ((u + 0.5) / out_width) * (2.0 * math.pi) - math.pi
    phi = phi + math.radians(lens.yaw_offset_deg)
    theta = ((v + 0.5) / out_height) * lens.theta_max_deg
    radial = np.asarray(lens.radius_at_theta(theta), dtype=np.float64)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    map_x = lens.center_x + radial[:, None] * sin_phi[None, :]
    map_y = lens.center_y - radial[:, None] * cos_phi[None, :]
    return (
        np.ascontiguousarray(map_x, dtype=np.float32),
        np.ascontiguousarray(map_y, dtype=np.float32),
    )


def load_lens_json(path: Path) -> LensModel:
    """Load a calibration file. Unknown keys are ignored aside from ``notes``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GeometryError(f"could not parse calibration file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GeometryError(f"calibration file {path} must contain a JSON object")
    unknown = sorted(set(data) - set(_CALIBRATION_FIELDS) - {"notes"})
    if unknown:
        log.warning("ignoring unknown calibration keys in %s: %s", path, ", ".join(unknown))
    kwargs: dict[str, float] = {}
    for key in _CALIBRATION_FIELDS:
        if key not in data:
            continue
        try:
            kwargs[key] = float(data[key])
        except (TypeError, ValueError) as exc:
            raise GeometryError(
                f"calibration field {key!r} in {path} is not a number"
            ) from exc
    return LensModel(**kwargs)


def save_lens_json(path: Path, lens: LensModel, notes: str | None = None) -> None:
    payload = lens.to_dict()
    if notes:
        payload_out: dict[str, object] = dict(payload)
        payload_out["notes"] = notes
    else:
        payload_out = dict(payload)
    path.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")


def autoscale_default_lens(lens: LensModel, width: int, height: int) -> LensModel:
    """Place the factory circle on a full sensor frame, a proxy, or a square window.

    A half-resolution 3072×1620 proxy of the 6144×3240 sensor keeps the circle
    centered if center and radius are halved. A square frame is the 250° circle's
    bounding window: on the sensor that circle is 3240 px across, so a 3240×3240
    extract is centered at (1620, 1620) with radius 1620. Explicit calibration
    is never passed through this function.
    """
    if width == SENSOR_WIDTH and height == SENSOR_HEIGHT:
        return lens
    if width <= 0 or height <= 0:
        raise GeometryError(f"invalid frame size {width}×{height}")
    if width == height:
        scale = width / float(SENSOR_HEIGHT)
        placed = replace(
            lens,
            center_x=width / 2.0,
            center_y=height / 2.0,
            radius=DEFAULT_RADIUS * scale,
        )
        log.info(
            "input is %dx%d, a square window of the 250° circle "
            "(that circle is %d px across on the KOMODO). "
            "Using center (%.1f, %.1f) and radius %.1f. Confirm with --calibrate.",
            width,
            height,
            SENSOR_HEIGHT,
            placed.center_x,
            placed.center_y,
            placed.radius,
        )
        return placed
    sx = width / SENSOR_WIDTH
    sy = height / SENSOR_HEIGHT
    if abs(sx - sy) > 1e-3:
        log.warning(
            "input is %dx%d, not a uniform scale of the KOMODO 6K %dx%d frame "
            "and not a square circle window; default center (%.1f, %.1f) and "
            "radius %.1f were left as the full-sensor values. Pass --center-x, "
            "--center-y, and --radius, or run --calibrate.",
            width,
            height,
            SENSOR_WIDTH,
            SENSOR_HEIGHT,
            lens.center_x,
            lens.center_y,
            lens.radius,
        )
        return lens
    log.info(
        "input is %dx%d (%.4f× the KOMODO 6K frame); scaling default center and radius",
        width,
        height,
        sx,
    )
    return lens.scaled(sx)


def lens_as_public_dict(lens: LensModel) -> dict[str, object]:
    """Calibration JSON plus the derived horizon radius, for sidecars."""
    payload: dict[str, object] = lens.to_dict()
    payload["horizon_radius_px"] = lens.radius_at_theta(HORIZON_THETA_DEG)
    payload["notes"] = (
        "Entaniya HAL 250 3.6mm on a zenith-mounted RED KOMODO 6K. "
        "r(θ) = radius * f(θ) / f(θ_max), f(θ) = θ_hat * (1 + k1 θ_hat^2 + "
        "k2 θ_hat^4 + k3 θ_hat^6), θ_hat = θ / θ_max."
    )
    return payload
