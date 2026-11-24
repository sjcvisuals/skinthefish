"""Polar mapping from an Entaniya HAL 250 zenith fisheye to equirectangular.

The lens is equidistant, ``r = f * θ``, and is mounted on a RED KOMODO 6K
pointing straight up. The optical axis is the zenith (polar angle 0). The
recorded circle covers 250° of field of view, which is polar angle 0° at the
center through 125° at the circle edge (35° below the horizon).

The delivery raster is a 16K-wide equirectangular image. Only the captured
125° strip is resampled; ``full-latlong`` mode pads the missing road-to-nadir
region with black so the file is a standard 2:1 LatLong.
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

# 16K UHD-aligned LatLong. Width is a multiple of 16. Full height is 2:1.
UHD_WIDTH = 15360
UHD_FULL_HEIGHT = 7680
FULL_LATLONG_VERTICAL_DEG = 180.0

# Working resolution of the unwrap, before the ~195.7% upscale.
# 15360 / 7850 = 1.9566..., and round(5333 * 7850 / 15360) = 2726.
NATIVE_WIDTH = 7850

OUTPUT_FULL_LATLONG = "full-latlong"
OUTPUT_ACTIVE_CROP = "active-crop"
OUTPUT_MODES = (OUTPUT_FULL_LATLONG, OUTPUT_ACTIVE_CROP)

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
        if not 0 < self.theta_max_deg <= FULL_LATLONG_VERTICAL_DEG:
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
    """Native unwrap size and the UHD delivery size derived from it."""

    native_width: int
    native_height: int
    delivery_width: int
    active_height: int
    full_height: int
    preview_scale: int
    theta_max_deg: float

    @property
    def pad_rows(self) -> int:
        return self.full_height - self.active_height

    @property
    def upscale_x(self) -> float:
        return self.delivery_width / self.native_width

    @property
    def upscale_y(self) -> float:
        return self.active_height / self.native_height

    def encoded_height(self, output_mode: str) -> int:
        if output_mode == OUTPUT_FULL_LATLONG:
            return self.full_height
        if output_mode == OUTPUT_ACTIVE_CROP:
            return self.active_height
        raise GeometryError(
            f"unknown output mode {output_mode!r}; expected one of {OUTPUT_MODES}"
        )


def master_active_height(theta_max_deg: float) -> int:
    """Rows of the 16K strip that cover ``theta_max_deg`` of a 180° LatLong."""
    if not 0 < theta_max_deg <= FULL_LATLONG_VERTICAL_DEG:
        raise GeometryError(
            f"theta_max_deg must be in (0, 180], got {theta_max_deg}"
        )
    active = round_half_up(UHD_FULL_HEIGHT * (theta_max_deg / FULL_LATLONG_VERTICAL_DEG))
    # Leave at least one row of pad when the capture is short of 180°.
    if theta_max_deg < FULL_LATLONG_VERTICAL_DEG:
        active = min(active, UHD_FULL_HEIGHT - 1)
    return max(2, active)


def delivery_raster(theta_max_deg: float = DEFAULT_THETA_MAX_DEG, preview_scale: int = 1) -> RasterSpec:
    """Build the unwrap and delivery sizes.

    Preview scales divide the master 16K raster. Scale 1 is the specified
    7850×2726 unwrap and the 15360×5333 / 15360×7680 delivery.
    """
    if preview_scale < 1:
        raise GeometryError(f"preview_scale must be >= 1, got {preview_scale}")
    active = master_active_height(theta_max_deg)
    native_h = round_half_up(active * NATIVE_WIDTH / UHD_WIDTH)
    native_w = NATIVE_WIDTH
    full_h = UHD_FULL_HEIGHT
    delivery_w = UHD_WIDTH
    if preview_scale > 1:
        native_w = max(16, round_half_up(native_w / preview_scale))
        native_h = max(8, round_half_up(native_h / preview_scale))
        delivery_w = max(16, round_half_up(delivery_w / preview_scale))
        active = max(8, round_half_up(active / preview_scale))
        full_h = max(active + 1, round_half_up(full_h / preview_scale))
    if delivery_w % 2:
        # 4:2:2 ProRes chroma is horizontal. Keep the encoded width even.
        delivery_w += 1
    if full_h <= active:
        full_h = active + 1
    if native_w < 16 or native_h < 8 or delivery_w < 16 or active < 8:
        raise GeometryError(
            f"preview-scale {preview_scale} collapses the raster below a usable size"
        )
    return RasterSpec(
        native_width=native_w,
        native_height=native_h,
        delivery_width=delivery_w,
        active_height=active,
        full_height=full_h,
        preview_scale=preview_scale,
        theta_max_deg=theta_max_deg,
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

    with the zenith at the top and ``θ_max`` at the bottom of the active strip.
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
    """Scale the factory KOMODO defaults when the file is a uniform proxy.

    A half-resolution 3072×1620 proxy of the same framing keeps the circle
    centered if center and radius are halved. Explicit calibration is never
    passed through this function.
    """
    if width == SENSOR_WIDTH and height == SENSOR_HEIGHT:
        return lens
    if width <= 0 or height <= 0:
        raise GeometryError(f"invalid frame size {width}×{height}")
    sx = width / SENSOR_WIDTH
    sy = height / SENSOR_HEIGHT
    if abs(sx - sy) > 1e-3:
        log.warning(
            "input is %dx%d, not a uniform scale of the KOMODO 6K %dx%d frame; "
            "default center and radius were left unscaled",
            width,
            height,
            SENSOR_WIDTH,
            SENSOR_HEIGHT,
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
