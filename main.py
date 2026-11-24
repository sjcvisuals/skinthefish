"""Unwrap Entaniya HAL 250 zenith fisheye plates to a 16:9 ProRes master.

The order of work is fixed: probe the source, decode ``yuv444p16le``, remap
the polar image to the native 16:9 strip, Lanczos-4 upscale that strip to
15360×8640 (``--size 16k``) or 7680×4320 (``--size 8k``), and encode the same
Apple ProRes profile the file arrived in.
Nothing is padded with black, and the samples are never converted to RGB.

Run ``python main.py --help`` for the command line. ``python gui.py`` opens
the small wrapper around the same options.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

from geometry import (
    DELIVERY_SIZES,
    HORIZON_THETA_DEG,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    GeometryError,
    LensModel,
    RasterSpec,
    autoscale_default_lens,
    delivery_raster,
    lens_as_public_dict,
    load_lens_json,
)
from media_io import (
    OUTPUT_PROFILE_CHOICES,
    ChunkedProResWriter,
    MediaError,
    MediaInfo,
    RawVideoReader,
    delivery_notes,
    encoder_pix_fmt,
    output_timecode,
    probe_media,
    profile_label,
    require_prores,
    resolve_frame_range,
    resolve_output_profile,
)
from resample import FrameResampler, cpu_unwrap, resolve_device
from upscaler import LanczosUpscaler, UpscaleError

log = logging.getLogger("hal250")

MOVIE_SUFFIXES = {".mov", ".mxf", ".mp4", ".qt"}
INTERPOLATION = {
    "lanczos4": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
}
__version__ = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skinthefish",
        description=(
            "Unwrap RED KOMODO 6K / Entaniya HAL 250 zenith fisheye footage "
            "into a 16:9 ProRes master. --size 16k is 15360×8640, --size 8k is "
            "7680×4320. The picture fills the frame. Color is not converted or sharpened."
        ),
    )
    parser.add_argument("input", nargs="?", type=Path, help="A movie, or a folder of movies.")
    parser.add_argument("-o", "--output", type=Path, help="Output movie. One input only.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for batch outputs. Defaults to each file's own folder.",
    )
    parser.add_argument("--calibration", type=Path, help="Lens JSON. CLI geometry flags override it.")
    parser.add_argument(
        "--center-x",
        type=float,
        help=(
            "Optical center X. Default 3072 on a 6144×3240 frame, "
            "or the frame center on a square circle window (1620 on 3240×3240)."
        ),
    )
    parser.add_argument(
        "--center-y",
        type=float,
        help="Optical center Y. Default 1620 on both the full sensor and a 3240×3240 window.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        help="Pixel radius of the 250° circle. Default 1620 on a full-resolution plate.",
    )
    parser.add_argument("--theta-max", type=float, help="Polar angle at the circle edge, degrees. Default 125.")
    parser.add_argument(
        "--yaw-offset",
        type=float,
        help="Degrees. Positive yaw turns the panorama clockwise (content moves left).",
    )
    parser.add_argument("--k1", type=float, help="Even polynomial coefficient on normalized polar angle.")
    parser.add_argument("--k2", type=float)
    parser.add_argument("--k3", type=float)
    parser.add_argument(
        "--interp",
        choices=tuple(INTERPOLATION),
        default="lanczos4",
        help="Kernel for the polar remap. The upscale is always Lanczos-4. Default lanczos4.",
    )
    parser.add_argument(
        "--size",
        choices=DELIVERY_SIZES,
        default="16k",
        help="Delivery raster. 16k is 15360×8640. 8k is 7680×4320. Same Lanczos scale either way.",
    )
    parser.add_argument(
        "--preview-scale",
        type=int,
        default=1,
        help="Divide the chosen 8K or 16K raster by this integer for a look-dev render. 1 is the master.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive first frame.")
    parser.add_argument("--end-frame", type=int, help="Exclusive end frame. Default: last frame.")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Write one overlay still (green 90° horizon, yellow 250° edge, center crosshair) and exit.",
    )
    parser.add_argument("--frame", type=int, help="Frame index for --calibrate. Default: the middle frame.")
    parser.add_argument("--calibrate-output", type=Path, help="PNG path for --calibrate.")
    parser.add_argument(
        "--profile",
        choices=OUTPUT_PROFILE_CHOICES,
        default="match",
        help="ProRes codec to write. match keeps the source profile. Default match.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Probe and print the plan. Do not decode.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses an NVIDIA GPU for the master resample when CUDA PyTorch is installed. The preview stays on the CPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="How many ProRes encodes to run at once. 0 uses one per CPU core. prores_ks does not fill extra cores inside one encode.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--gui", action="store_true", help="Open the lightweight wrapper.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"skinthefish {__version__}")
    return parser


def configure_logging(verbose: bool) -> None:
    logger = logging.getLogger("hal250")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    cv2.setNumThreads(os.cpu_count() or 1)
    try:
        if args.gui or (args.input is None and argv is None):
            from studio import serve

            serve(initial_path=args.input)
            return 0
        return run(args)
    except (GeometryError, MediaError, UpscaleError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    if args.input is None:
        raise MediaError("pass an input movie or folder (or --gui)")
    _ensure_tools(args.ffmpeg, args.ffprobe)
    sources = collect_inputs(args.input)
    if args.calibrate:
        for index, source in enumerate(sources, start=1):
            log.info("calibrate %d/%d %s", index, len(sources), source)
            calibrate_file(args, source, len(sources))
        return 0
    for index, source in enumerate(sources, start=1):
        output = movie_output_path(source, args, len(sources))
        log.info("render %d/%d %s", index, len(sources), source)
        render_movie(args, source, output)
    return 0


def render_movie(args: argparse.Namespace, source: Path, output: Path) -> None:
    info = probe_media(source, args.ffprobe)
    require_prores(info)
    for note in delivery_notes(info):
        log.info("%s", note)
    lens = resolve_lens(args, info.width, info.height)
    raster = delivery_raster(lens.theta_max_deg, args.preview_scale, args.size)
    start, end, count = resolve_frame_range(info, args.start_frame, args.end_frame)
    timecode = output_timecode(info, start)
    output_profile = resolve_output_profile(getattr(args, "profile", "match"), info.profile)
    whole_file = start == 0 and (end is None or end == info.frame_count)
    if output.resolve() == source.resolve():
        raise MediaError(f"output path is the same as the input: {output}")
    log.info(
        "\n%s",
        describe_job(
            info,
            lens,
            raster,
            output,
            args.interp,
            start,
            end,
            timecode,
            output_profile,
        ),
    )
    if args.dry_run:
        return
    device = resolve_device(getattr(args, "device", "auto"), raster.delivery_width * raster.delivery_height)
    resampler = FrameResampler(lens, raster, INTERPOLATION[args.interp], device)
    log.info("resample: %s", resampler.label)
    previous_threads = cv2.getNumThreads()
    # One ProRes process barely uses extra cores. Leave half of them for the
    # resample and give the rest to the parallel encodes.
    cv2.setNumThreads(max(1, (os.cpu_count() or 1) // 2))
    started = time.perf_counter()
    try:
        progress = Progress(count, getattr(args, "on_progress", None))
        with RawVideoReader(
            source,
            info,
            ffmpeg=args.ffmpeg,
            start_frame=0 if whole_file else start,
            end_frame=None if whole_file else end,
        ) as reader, ChunkedProResWriter(
            output,
            source=source,
            width=raster.delivery_width,
            height=raster.delivery_height,
            fps=info.fps,
            profile=output_profile,
            info=info,
            timecode=timecode,
            start_frame=0 if whole_file else start,
            frame_count=None if whole_file else count,
            ffmpeg=args.ffmpeg,
            overwrite=args.overwrite,
            workers=getattr(args, "workers", 0),
            on_encoded=progress.advance,
        ) as writer:
            for frame in reader.frames():
                writer.write(resampler.apply(frame))
        progress.close()
    finally:
        cv2.setNumThreads(previous_threads)
    elapsed = time.perf_counter() - started
    log.info("wrote %s (%d frames, %.1fs)", output, writer.frames_written, elapsed)


def calibrate_file(args: argparse.Namespace, source: Path, count: int) -> None:
    info = probe_media(source, args.ffprobe)
    lens = resolve_lens(args, info.width, info.height)
    frame_index = calibration_frame_index(info, args.frame)
    output = calibration_output_path(source, args, count)
    log.info(
        "calibration frame %d of %s -> %s\ncenter (%.3f, %.3f) radius %.3f horizon %.3f yaw %.3f",
        frame_index,
        source,
        output,
        lens.center_x,
        lens.center_y,
        lens.radius,
        float(lens.radius_at_theta(HORIZON_THETA_DEG)),
        lens.yaw_offset_deg,
    )
    if args.dry_run:
        return
    if output.exists() and not args.overwrite:
        raise MediaError(f"calibration still exists: {output} (pass --overwrite)")
    sidecar = output.with_suffix(".json")
    if sidecar.exists() and not args.overwrite:
        raise MediaError(f"calibration sidecar exists: {sidecar} (pass --overwrite)")
    frame = read_one_frame(source, info, frame_index, args.ffmpeg, layout="rgb")
    image = draw_calibration_overlay(frame, lens)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise MediaError(f"could not write {output}")
    payload = lens_as_public_dict(lens)
    payload["source"] = str(source)
    payload["frame"] = frame_index
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s and %s", output, sidecar)


def unwrap_and_upscale(
    frame: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray],
    upscaler: LanczosUpscaler,
    raster: RasterSpec,
    interpolation: int,
) -> np.ndarray:
    """Remap at the native 16:9 size, then Lanczos-4 upscale to the delivery size.

    Samples that fall outside the source picture are filled with code 0. That
    is only the missing texels. The frame itself is not padded, and no
    contrast or color operator runs after the resample.
    """
    return cpu_unwrap(frame, maps, upscaler, raster, interpolation)


def draw_calibration_overlay(frame: np.ndarray, lens: LensModel) -> np.ndarray:
    """BGR preview with the horizon, the 250° edge, and the optical center.

    ``F`` marks the sensor direction that lands on the panorama center for the
    current yaw. ``R``, ``B``, and ``L`` step clockwise from it.
    """
    if frame.dtype != np.uint16:
        raise MediaError(f"calibration frame is {frame.dtype}, expected uint16")
    preview = (frame >> 8).astype(np.uint8)
    bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    dist = np.sqrt((xs - np.float32(lens.center_x)) ** 2 + (ys - np.float32(lens.center_y)) ** 2)
    horizon_r = float(lens.radius_at_theta(HORIZON_THETA_DEG))
    edge_r = float(lens.radius_at_theta(lens.theta_max_deg))
    ring = max(1.0, float(lens.radius) / 900.0)
    bgr[np.abs(dist - horizon_r) <= ring] = (0, 220, 0)
    bgr[np.abs(dist - edge_r) <= ring] = (0, 220, 255)
    arm = max(0.6, width / 5000.0)
    columns = np.abs(np.arange(width, dtype=np.float32) - np.float32(lens.center_x)) <= arm
    rows = np.abs(np.arange(height, dtype=np.float32) - np.float32(lens.center_y)) <= arm
    bgr[:, columns] = (255, 255, 255)
    bgr[rows, :] = (255, 255, 255)
    _draw_heading_ticks(bgr, lens, horizon_r)
    _draw_legend(bgr, lens, horizon_r, edge_r)
    return bgr


def describe_job(
    info: MediaInfo,
    lens: LensModel,
    raster: RasterSpec,
    output: Path,
    interpolation: str,
    start_frame: int,
    end_frame: int | None,
    timecode: str | None,
    output_profile: int,
) -> str:
    profile = profile_label(output_profile)
    pix = encoder_pix_fmt(output_profile)
    rate = f"{info.fps.numerator}/{info.fps.denominator}"
    end_text = "end" if end_frame is None else str(end_frame)
    lines = [
        f"input {info.path}",
        (
            f"  {info.width}×{info.height}  {rate} fps  "
            f"frames {info.frame_count if info.frame_count is not None else '?'}  "
            f"range [{start_frame}, {end_text})  {info.profile_label}  {info.pix_fmt or '?'}"
        ),
        (
            f"  color primaries={info.color_primaries or '-'} "
            f"transfer={info.color_transfer or '-'} "
            f"matrix={info.color_space or '-'} range={info.color_range or '-'}  "
            f"timecode {timecode or '-'}  audio={'yes' if info.has_audio else 'no'}"
        ),
        (
            f"  lens center ({lens.center_x:.3f}, {lens.center_y:.3f})  "
            f"radius {lens.radius:.3f}  horizon {float(lens.radius_at_theta(HORIZON_THETA_DEG)):.3f}  "
            f"theta_max {lens.theta_max_deg:.3f}°  yaw {lens.yaw_offset_deg:.3f}°  "
            f"k [{lens.k1:g}, {lens.k2:g}, {lens.k3:g}]"
        ),
        (
            f"  unwrap {raster.native_width}×{raster.native_height}  remap {interpolation}  "
            f"upscale Lanczos-4 {raster.upscale_x * 100:.2f}% × {raster.upscale_y * 100:.2f}%"
        ),
        (
            f"  output {output}  {raster.size}  {raster.delivery_width}×{raster.delivery_height}  16:9  "
            f"no pad  profile {profile}  {pix}  vendor apl0  yuv444p16 working"
        ),
    ]
    return "\n".join(lines)


def resolve_lens(args: argparse.Namespace, width: int, height: int) -> LensModel:
    spatial = args.calibration is not None
    lens = load_lens_json(args.calibration) if args.calibration else LensModel()
    updates: dict[str, float] = {}
    for attr, field in (
        ("center_x", "center_x"),
        ("center_y", "center_y"),
        ("radius", "radius"),
        ("theta_max", "theta_max_deg"),
        ("yaw_offset", "yaw_offset_deg"),
        ("k1", "k1"),
        ("k2", "k2"),
        ("k3", "k3"),
    ):
        value = getattr(args, attr)
        if value is not None:
            updates[field] = float(value)
    if any(getattr(args, name) is not None for name in ("center_x", "center_y", "radius")):
        spatial = True
    if updates:
        lens = replace(lens, **updates)
    if spatial:
        if (width, height) != (SENSOR_WIDTH, SENSOR_HEIGHT):
            log.warning(
                "frame is %dx%d and the lens geometry was set explicitly; "
                "center and radius are used as given, not scaled from the %dx%d sensor",
                width,
                height,
                SENSOR_WIDTH,
                SENSOR_HEIGHT,
            )
        return lens
    return autoscale_default_lens(lens, width, height)


def collect_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in MOVIE_SUFFIXES
        )
        if not files:
            raise MediaError(f"no .mov, .mxf, .mp4, or .qt files in {path}")
        return files
    if not path.is_file():
        raise MediaError(f"input does not exist: {path}")
    return [path]


def movie_output_path(source: Path, args: argparse.Namespace, count: int) -> Path:
    if args.output is not None:
        if count > 1:
            raise MediaError("use --output-dir when rendering more than one movie")
        return args.output
    directory = args.output_dir if args.output_dir is not None else source.parent
    suffix = f"_{args.size}"
    if args.preview_scale != 1:
        suffix += f"_preview{args.preview_scale}"
    return directory / f"{source.stem}{suffix}.mov"


def calibration_output_path(source: Path, args: argparse.Namespace, count: int) -> Path:
    if args.calibrate_output is not None:
        if count > 1:
            raise MediaError(
                "--calibrate-output names one still; omit it to write <clip>_calibration.png per file"
            )
        return args.calibrate_output
    directory = args.output_dir if args.output_dir is not None else source.parent
    return directory / f"{source.stem}_calibration.png"


def calibration_frame_index(info: MediaInfo, requested: int | None) -> int:
    if requested is not None:
        if requested < 0:
            raise MediaError(f"--frame must be >= 0, got {requested}")
        if info.frame_count is not None and requested >= info.frame_count:
            raise MediaError(
                f"--frame {requested} is past the end of a {info.frame_count}-frame clip"
            )
        return requested
    if info.frame_count is None or info.frame_count <= 1:
        return 0
    return info.frame_count // 2


def read_one_frame(
    source: Path,
    info: MediaInfo,
    frame_index: int,
    ffmpeg: str,
    layout: str = "yuv",
) -> np.ndarray:
    with RawVideoReader(
        source,
        info,
        ffmpeg=ffmpeg,
        start_frame=frame_index,
        end_frame=frame_index + 1,
        layout=layout,
    ) as reader:
        for frame in reader.frames():
            return np.array(frame, copy=True)
    raise MediaError(f"could not read frame {frame_index} from {source}")


def _draw_heading_ticks(bgr: np.ndarray, lens: LensModel, horizon_r: float) -> None:
    tick = max(8.0, horizon_r * 0.035)
    labels = (("F", 0.0, (0, 255, 0)), ("R", 90.0, (255, 255, 255)), ("B", 180.0, (255, 255, 255)), ("L", 270.0, (255, 255, 255)))
    font_scale = max(0.4, bgr.shape[1] / 3500.0)
    for name, bearing, color in labels:
        phi = math.radians(bearing + lens.yaw_offset_deg)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        inner = horizon_r - tick
        outer = horizon_r + tick
        p0 = (
            int(round(lens.center_x + inner * sin_phi)),
            int(round(lens.center_y - inner * cos_phi)),
        )
        p1 = (
            int(round(lens.center_x + outer * sin_phi)),
            int(round(lens.center_y - outer * cos_phi)),
        )
        cv2.line(bgr, p0, p1, color, thickness=max(1, int(round(font_scale * 2))), lineType=cv2.LINE_AA)
        label_r = horizon_r + tick * 2.2
        origin = (
            int(round(lens.center_x + label_r * sin_phi)),
            int(round(lens.center_y - label_r * cos_phi)),
        )
        cv2.putText(
            bgr,
            name,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness=max(1, int(round(font_scale * 2))),
            lineType=cv2.LINE_AA,
        )


def _draw_legend(bgr: np.ndarray, lens: LensModel, horizon_r: float, edge_r: float) -> None:
    lines = [
        "HAL 250 calibration",
        f"center {lens.center_x:.2f}, {lens.center_y:.2f}",
        f"radius {lens.radius:.2f}   horizon {horizon_r:.2f}",
        f"theta max {lens.theta_max_deg:.2f}   yaw {lens.yaw_offset_deg:.2f}",
        f"edge radius {edge_r:.2f}   F = pano center",
    ]
    scale = max(0.45, min(1.2, bgr.shape[1] / 2200.0))
    thickness = max(1, int(round(scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_h = max(size[1] for size in sizes) + 8
    box_w = max(size[0] for size in sizes) + 20
    box_h = line_h * len(lines) + 12
    cv2.rectangle(bgr, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), thickness=-1)
    for index, line in enumerate(lines):
        origin = (18, 8 + (index + 1) * line_h)
        cv2.putText(bgr, line, origin, font, scale, (240, 240, 240), thickness, cv2.LINE_AA)


def _validate_args(args: argparse.Namespace) -> None:
    if args.preview_scale < 1:
        raise GeometryError("--preview-scale must be >= 1")
    if args.input is None and not args.gui:
        raise MediaError("pass an input movie or folder")


def _ensure_tools(ffmpeg: str, ffprobe: str) -> None:
    for binary in (ffmpeg, ffprobe):
        if Path(binary).is_file() or shutil.which(binary):
            continue
        raise MediaError(f"could not find {binary!r}. Install FFmpeg 6 or newer and put it on PATH.")


class Progress:
    def __init__(self, total: int | None, callback=None) -> None:
        self.total = total
        self.count = 0
        self.started = time.perf_counter()
        self.callback = callback

    def update(self) -> None:
        self.advance(1)

    def advance(self, count: int) -> None:
        self.count += count
        elapsed = time.perf_counter() - self.started
        rate = self.count / elapsed if elapsed else 0.0
        if self.total:
            remaining = max(0, self.total - self.count)
            eta = remaining / rate if rate else 0.0
            message = f"frame {self.count}/{self.total}  {rate:.2f} fps  eta {eta:.0f}s"
        else:
            message = f"frame {self.count}  {rate:.2f} fps"
        print("\r" + message, end="", file=sys.stderr, flush=True)
        if self.callback is not None:
            self.callback(self.count, self.total, message)

    def close(self) -> None:
        if self.count:
            print(file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
