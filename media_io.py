"""FFmpeg probe, 16-bit YUV decode, and ProRes encode that keeps the source flavor.

Frames stay in ``yuv444p16le``. That is a bit-depth and chroma upsample of the
source YUV, not a conversion to RGB, so primaries, transfer, and code values
are not reinterpreted. The encoder is ``prores_ks`` with the source profile,
``-vendor apl0``, and the same color tags the file arrived with.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import logging
from pathlib import Path
import subprocess
import threading

import numpy as np

log = logging.getLogger("hal250.media")

# Working format. 4:4:4 16-bit YUV is a higher-precision copy of the source
# codes. It is not an RGB conversion.
PROCESS_PIX_FMT = "yuv444p16le"

# prores_ks stores width and height in the frame header as uint16.
PRORES_MAX_DIMENSION = 65535

# Apple ProRes fourccs. These are more reliable than the profile name.
_TAG_TO_PROFILE: dict[str, tuple[int, str]] = {
    "apco": (0, "proxy"),
    "apcs": (1, "lt"),
    "apcn": (2, "standard"),
    "apch": (3, "hq"),
    "ap4h": (4, "4444"),
    "ap4x": (5, "4444xq"),
}

_NAME_TO_PROFILE: dict[str, tuple[int, str]] = {
    "proxy": (0, "proxy"),
    "lt": (1, "lt"),
    "standard": (2, "standard"),
    "hq": (3, "hq"),
    "4444": (4, "4444"),
    "xq": (5, "4444xq"),
    "4444xq": (5, "4444xq"),
    "4444 xq": (5, "4444xq"),
}

_PROFILE_LABEL = {
    0: "ProRes 422 Proxy",
    1: "ProRes 422 LT",
    2: "ProRes 422",
    3: "ProRes 422 HQ",
    4: "ProRes 4444",
    5: "ProRes 4444 XQ",
}

_UNSPECIFIED_COLOR = {"", "unknown", "unspecified", "reserved"}


class MediaError(RuntimeError):
    """Probe, decode, or encode failed."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frame_count: int | None
    duration: float | None
    timecode: str | None
    drop_frame: bool
    pix_fmt: str | None
    profile: int | None
    profile_name: str | None
    codec_tag: str | None
    codec_name: str
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    color_range: str | None
    bits_per_raw_sample: int | None
    has_audio: bool
    audio_streams: int
    has_alpha: bool

    @property
    def is_prores(self) -> bool:
        return self.codec_name == "prores" and self.profile is not None

    @property
    def profile_label(self) -> str:
        if self.profile is None:
            return self.codec_name
        return _PROFILE_LABEL.get(self.profile, self.profile_name or "prores")


def encoder_pix_fmt(profile: int) -> str:
    """Pixel format passed to ``prores_ks``.

    4444 and 4444 XQ are encoded as ``yuv444p10le``. FFmpeg's decoder then
    reports those files as ``yuv444p12le`` because the ProRes 4444 codeword
    is 12 bits wide; ``prores_ks`` itself only accepts the 10-bit formats.
    """
    if profile >= 4:
        return "yuv444p10le"
    return "yuv422p10le"


def delivery_notes(info: MediaInfo) -> list[str]:
    """Honest notes about what the ProRes writer can and cannot preserve."""
    notes: list[str] = []
    if info.has_alpha:
        notes.append(
            "source has an alpha plane; the unwrap stays in YUV and the alpha is discarded"
        )
    if info.profile is not None and info.profile >= 4:
        notes.append(
            "ProRes 4444 is written with prores_ks as yuv444p10le (10-bit samples). "
            "ffprobe reports the 4444 codeword as 12-bit yuv444p12le; the extra two "
            "bits are not stored by this encoder"
        )
    return notes


def probe_media(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    """Read resolution, rate, ProRes profile, color, and timecode."""
    if not path.is_file():
        raise MediaError(f"input does not exist: {path}")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise MediaError(
            f"could not run {ffprobe!r}. Install FFmpeg and make sure ffprobe is on PATH."
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "ffprobe failed"
        raise MediaError(f"ffprobe failed for {path}: {detail}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path}") from exc
    return parse_probe_json(payload, path)


def parse_probe_json(payload: dict, path: Path) -> MediaInfo:
    """Turn an ffprobe JSON document into :class:`MediaInfo`."""
    streams = payload.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise MediaError(f"no video stream in {path}")
    if len(video_streams) > 1:
        log.warning("%s has %d video streams; using the first", path, len(video_streams))
    video = video_streams[0]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    codec_name = str(video.get("codec_name") or "")
    tag = str(video.get("codec_tag_string") or "").lower()
    profile_name_raw = str(video.get("profile") or "")
    profile: int | None
    profile_name: str | None
    if tag in _TAG_TO_PROFILE:
        profile, profile_name = _TAG_TO_PROFILE[tag]
    else:
        mapped = _NAME_TO_PROFILE.get(profile_name_raw.strip().lower())
        if mapped and codec_name == "prores":
            profile, profile_name = mapped
        else:
            profile, profile_name = None, None

    fps = _parse_fps(video)
    frame_count = _parse_frame_count(video, payload.get("format") or {}, fps)
    timecode, drop_frame = _parse_timecode(video, streams, payload.get("format") or {})
    pix_fmt = video.get("pix_fmt") or None
    has_alpha = bool(pix_fmt and ("yuva" in pix_fmt or "alpha" in pix_fmt))
    bits = video.get("bits_per_raw_sample")
    try:
        bits_i = int(bits) if bits not in (None, "N/A", "") else None
    except (TypeError, ValueError):
        bits_i = None

    try:
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError(f"video stream in {path} has no width/height") from exc

    duration = _optional_float((payload.get("format") or {}).get("duration"))
    if duration is None:
        duration = _optional_float(video.get("duration"))

    return MediaInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        timecode=timecode,
        drop_frame=drop_frame,
        pix_fmt=pix_fmt,
        profile=profile,
        profile_name=profile_name,
        codec_tag=tag or None,
        codec_name=codec_name,
        color_primaries=_known_color(video.get("color_primaries")),
        color_transfer=_known_color(video.get("color_transfer")),
        color_space=_known_color(video.get("color_space")),
        color_range=_known_range(video.get("color_range")),
        bits_per_raw_sample=bits_i,
        has_audio=bool(audio),
        audio_streams=len(audio),
        has_alpha=has_alpha,
    )


def check_prores_dimensions(width: int, height: int) -> None:
    """Reject sizes ``prores_ks`` cannot store, and warn on macroblock padding."""
    if width < 2 or height < 2:
        raise MediaError(f"ProRes frame is too small: {width}×{height}")
    if width > PRORES_MAX_DIMENSION or height > PRORES_MAX_DIMENSION:
        raise MediaError(
            "ProRes stores width and height as 16-bit fields in the frame header, "
            f"so neither side can exceed {PRORES_MAX_DIMENSION}. "
            f"Requested {width}×{height}. "
            "Use --preview-scale or a smaller delivery raster."
        )
    if width % 2:
        raise MediaError(
            f"ProRes 4:2:2 needs an even width, got {width}. "
            "The 16K delivery width 15360 is valid."
        )
    if width % 16 or height % 16:
        log.warning(
            "delivery %dx%d is not a multiple of 16. prores_ks pads slices "
            "internally and still stores %dx%d. The 15360×8640 master is aligned.",
            width,
            height,
            width,
            height,
        )


def nominal_timecode_fps(fps: Fraction) -> int:
    """Timecode counts at the rounded integer rate (23.976 -> 24, 29.97 -> 30)."""
    nominal = int(round(float(fps)))
    if nominal <= 0:
        raise MediaError(f"cannot build a timecode from frame rate {fps}")
    return nominal


def parse_timecode(value: str, drop_frame: bool | None = None) -> tuple[int, int, int, int, bool]:
    """Parse ``HH:MM:SS:FF`` or ``HH:MM:SS;FF``. Returns hours, minutes, seconds, frames, drop."""
    text = value.strip()
    if len(text) < 11 or text[2] != ":" or text[5] != ":":
        raise MediaError(f"unrecognized timecode {value!r}")
    sep = text[8]
    if sep not in ":;":
        raise MediaError(f"unrecognized timecode {value!r}")
    try:
        hours = int(text[0:2])
        minutes = int(text[3:5])
        seconds = int(text[6:8])
        frames = int(text[9:11])
    except ValueError as exc:
        raise MediaError(f"unrecognized timecode {value!r}") from exc
    drop = sep == ";" if drop_frame is None else drop_frame
    return hours, minutes, seconds, frames, drop


def format_timecode(hours: int, minutes: int, seconds: int, frames: int, drop_frame: bool) -> str:
    sep = ";" if drop_frame else ":"
    return f"{hours % 24:02d}:{minutes:02d}:{seconds:02d}{sep}{frames:02d}"


def shift_timecode(timecode: str, frame_offset: int, fps: Fraction, drop_frame: bool) -> str:
    """Move a timecode forward by ``frame_offset`` frames at ``fps``."""
    if frame_offset == 0:
        _h, _m, _s, _f, parsed_drop = parse_timecode(timecode, drop_frame)
        return format_timecode(_h, _m, _s, _f, parsed_drop and drop_frame)
    nominal = nominal_timecode_fps(fps)
    hours, minutes, seconds, frames, parsed_drop = parse_timecode(timecode, drop_frame)
    drop = parsed_drop and drop_frame
    if drop and nominal not in (30, 60):
        log.warning(
            "drop-frame timecode is defined for 29.97 and 59.94; writing non-drop at %s fps",
            nominal,
        )
        drop = False
    absolute = _timecode_to_frames(hours, minutes, seconds, frames, nominal, drop)
    absolute += frame_offset
    h, m, s, f = _frames_to_timecode(absolute, nominal, drop)
    return format_timecode(h, m, s, f, drop)


class RawVideoReader:
    """Stream frames from ``ffmpeg`` without writing a temp file.

    ``layout="yuv"`` is the master path: planar ``yuv444p16le`` interleaved to
    ``H×W×3`` as Y, U, V. ``layout="rgb"`` is only for the calibration PNG,
    which is a viewing still and is never encoded back.
    """

    def __init__(
        self,
        path: Path,
        info: MediaInfo,
        *,
        ffmpeg: str = "ffmpeg",
        start_frame: int = 0,
        end_frame: int | None = None,
        layout: str = "yuv",
    ) -> None:
        if layout not in {"yuv", "rgb"}:
            raise MediaError(f"unknown frame layout {layout!r}")
        self.path = path
        self.info = info
        self.ffmpeg = ffmpeg
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.layout = layout
        self.proc: subprocess.Popen[bytes] | None = None
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self.frame_bytes = info.width * info.height * 3 * 2

    def __enter__(self) -> RawVideoReader:
        self.proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.proc.stderr is not None
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self.proc.stderr, self._stderr), daemon=True
        )
        self._stderr_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        self.proc = None

    def frames(self):
        """Yield one ``uint16`` ``H×W×3`` frame per decoded picture."""
        if self.proc is None or self.proc.stdout is None:
            raise MediaError("decoder is not running")
        height, width = self.info.height, self.info.width
        while True:
            raw = _read_exact(self.proc.stdout, self.frame_bytes)
            if raw is None:
                break
            if self.layout == "rgb":
                yield np.frombuffer(raw, dtype="<u2").reshape(height, width, 3)
                continue
            planes = np.frombuffer(raw, dtype="<u2").reshape(3, height, width)
            yield np.stack((planes[0], planes[1], planes[2]), axis=-1)
        code = self.proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        if code != 0:
            detail = self._stderr.decode("utf-8", errors="replace").strip()
            raise MediaError(f"ffmpeg decode failed ({code}): {detail or 'no stderr'}")

    def _command(self) -> list[str]:
        cmd = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.path),
            "-map",
            "0:v:0",
        ]
        trim = _trim_filter(self.start_frame, self.end_frame)
        if trim:
            cmd += ["-vf", trim]
        pix_fmt = "rgb48le" if self.layout == "rgb" else PROCESS_PIX_FMT
        cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
        log.debug("decode command: %s", _format_cmd(cmd))
        return cmd


class ProResWriter:
    """Pipe ``yuv444p16le`` frames into ``prores_ks`` and mux source audio back in."""

    def __init__(
        self,
        output: Path,
        *,
        source: Path,
        width: int,
        height: int,
        fps: Fraction,
        profile: int,
        info: MediaInfo,
        timecode: str | None,
        start_frame: int,
        frame_count: int | None,
        ffmpeg: str = "ffmpeg",
        overwrite: bool = False,
    ) -> None:
        check_prores_dimensions(width, height)
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.profile = profile
        self.pix_fmt = encoder_pix_fmt(profile)
        self.ffmpeg = ffmpeg
        self.overwrite = overwrite
        self.source = source
        self.info = info
        self.timecode = timecode
        self.start_frame = start_frame
        self.frame_count = frame_count
        self.proc: subprocess.Popen[bytes] | None = None
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self.frames_written = 0

    def __enter__(self) -> ProResWriter:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists() and not self.overwrite:
            raise MediaError(
                f"output already exists: {self.output} (pass --overwrite to replace it)"
            )
        try:
            self.proc = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise MediaError(
                f"could not run {self.ffmpeg!r}. Install FFmpeg and make sure it is on PATH."
            ) from exc
        assert self.proc.stderr is not None
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self.proc.stderr, self._stderr), daemon=True
        )
        self._stderr_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
            return
        self.close()

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise MediaError("encoder is not running")
        if frame.dtype != np.uint16 or frame.shape != (self.height, self.width, 3):
            raise MediaError(
                f"encoder expected uint16 YUV {self.width}×{self.height}, "
                f"got {frame.dtype} {getattr(frame, 'shape', None)}"
            )
        payload = interleaved_to_planar_bytes(frame)
        try:
            self.proc.stdin.write(payload)
        except BrokenPipeError as exc:
            self.proc.wait()
            detail = self._stderr_text()
            raise MediaError(self._encode_failure_message(self.proc.returncode or 1, detail)) from exc
        self.frames_written += 1

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.stdin is not None:
            proc.stdin.close()
        code = proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        self.proc = None
        if code != 0:
            raise MediaError(self._encode_failure_message(code, self._stderr_text()))

    def abort(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        self.proc = None

    def _command(self) -> list[str]:
        rate = f"{self.fps.numerator}/{self.fps.denominator}"
        cmd = [
            self.ffmpeg,
            "-y" if self.overwrite else "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            PROCESS_PIX_FMT,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            rate,
        ]
        # Tag the pipe with the source color description so swscale does not
        # invent a matrix on the way back to 10-bit ProRes.
        cmd += _color_flag_args(self.info)
        cmd += ["-i", "pipe:0"]
        if self.start_frame or self.frame_count is not None:
            start = Fraction(self.start_frame) / self.fps
            cmd += ["-ss", _format_seconds(start)]
            if self.frame_count is not None:
                duration = Fraction(self.frame_count) / self.fps
                cmd += ["-t", _format_seconds(duration)]
        cmd += ["-i", str(self.source)]
        cmd += [
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map_metadata",
            "1:g",
            "-c:v",
            "prores_ks",
            "-profile:v",
            str(self.profile),
            "-pix_fmt",
            self.pix_fmt,
            "-vendor",
            "apl0",
            "-alpha_bits",
            "0",
            "-c:a",
            "copy",
        ]
        if self.timecode:
            cmd += ["-timecode", self.timecode]
        for flag, value in _color_output_flags(self.info):
            cmd += [flag, value]
        cmd.append(str(self.output))
        log.debug("encode command: %s", _format_cmd(cmd))
        return cmd

    def _stderr_text(self) -> str:
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        return self._stderr.decode("utf-8", errors="replace").strip()

    def _encode_failure_message(self, code: int, detail: str) -> str:
        hint = ""
        if self.width > 8192 or self.height > 8192:
            hint = (
                " This FFmpeg build's prores_ks may be rejecting a frame above 8K. "
                f"Requested {self.width}×{self.height}. "
                "Width 15360 was checked against FFmpeg 6.1 prores_ks; if this "
                "binary is older, upgrade FFmpeg or render with --preview-scale."
            )
        text = detail or "no stderr"
        return f"ffmpeg encode failed ({code}): {text}.{hint}"


def require_prores(info: MediaInfo) -> None:
    if info.is_prores and info.profile is not None:
        return
    raise MediaError(
        f"{info.path} is {info.codec_name or 'not ProRes'}"
        + (f" ({info.pix_fmt})" if info.pix_fmt else "")
        + ". This tool matches the source Apple ProRes profile, so the input "
        "needs to be ProRes (422 Proxy, LT, 422, HQ, 4444, or 4444 XQ). "
        "Transcode to ProRes first, then unwrap."
    )


def resolve_frame_range(
    info: MediaInfo, start_frame: int, end_frame: int | None
) -> tuple[int, int | None, int | None]:
    """Return ``(start, end, count)`` with ``end`` exclusive and ``count`` optional."""
    if start_frame < 0:
        raise MediaError(f"--start-frame must be >= 0, got {start_frame}")
    if info.frame_count is not None and start_frame > info.frame_count:
        raise MediaError(
            f"--start-frame {start_frame} is past the end of a {info.frame_count}-frame clip"
        )
    end = end_frame
    if end is None and info.frame_count is not None:
        end = info.frame_count
    if end is not None and end < start_frame:
        raise MediaError(f"--end-frame {end} is before --start-frame {start_frame}")
    if end is not None and info.frame_count is not None and end > info.frame_count:
        log.warning(
            "--end-frame %d is past the probed length %d; clamping",
            end,
            info.frame_count,
        )
        end = info.frame_count
    count = None if end is None else end - start_frame
    if count == 0:
        raise MediaError("frame range is empty")
    return start_frame, end, count


def output_timecode(info: MediaInfo, start_frame: int) -> str | None:
    if not info.timecode:
        return None
    return shift_timecode(info.timecode, start_frame, info.fps, info.drop_frame)


def _parse_fps(video: dict) -> Fraction:
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = _rate(video.get(key))
        if rate is not None:
            other_key = "r_frame_rate" if key == "avg_frame_rate" else "avg_frame_rate"
            other = _rate(video.get(other_key))
            if other is not None and other != rate:
                log.warning(
                    "avg_frame_rate %s and r_frame_rate %s differ; using %s",
                    video.get("avg_frame_rate"),
                    video.get("r_frame_rate"),
                    rate,
                )
            return rate
    raise MediaError("video stream has no frame rate")


def _rate(value: object) -> Fraction | None:
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    if rate <= 0:
        return None
    return rate


def _parse_frame_count(video: dict, fmt: dict, fps: Fraction) -> int | None:
    raw = video.get("nb_frames")
    if raw not in (None, "N/A", ""):
        try:
            count = int(raw)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count
    duration = _optional_float(video.get("duration"))
    if duration is None:
        duration = _optional_float(fmt.get("duration"))
    if duration is None or duration <= 0:
        return None
    return int(round(duration * float(fps)))


def _parse_timecode(video: dict, streams: list, fmt: dict) -> tuple[str | None, bool]:
    candidates: list[str] = []
    tags = video.get("tags") or {}
    if tags.get("timecode"):
        candidates.append(str(tags["timecode"]))
    for stream in streams:
        if str(stream.get("codec_tag_string") or "").lower() != "tmcd":
            continue
        tc = (stream.get("tags") or {}).get("timecode")
        if tc:
            candidates.append(str(tc))
    fmt_tc = (fmt.get("tags") or {}).get("timecode")
    if fmt_tc:
        candidates.append(str(fmt_tc))
    if not candidates:
        return None, False
    timecode = candidates[0].strip()
    drop = ";" in timecode
    return timecode, drop


def _known_color(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _UNSPECIFIED_COLOR:
        return None
    return text


def _known_range(value: object) -> str | None:
    text = _known_color(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"tv", "mpeg", "limited"}:
        return "tv"
    if lowered in {"pc", "jpeg", "full"}:
        return "pc"
    return None


def interleaved_to_planar_bytes(frame: np.ndarray) -> bytes:
    """Pack ``H×W×3`` YUV back into ffmpeg's planar ``yuv444p16le`` order."""
    return b"".join(np.ascontiguousarray(frame[:, :, channel]).tobytes() for channel in range(3))


def _color_flag_args(info: MediaInfo) -> list[str]:
    args: list[str] = []
    for flag, value in _color_output_flags(info):
        args.extend((flag, value))
    return args


def _color_output_flags(info: MediaInfo) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    if info.color_primaries:
        flags.append(("-color_primaries", info.color_primaries))
    if info.color_transfer:
        flags.append(("-color_trc", info.color_transfer))
    if info.color_space:
        flags.append(("-colorspace", info.color_space))
    if info.color_range:
        flags.append(("-color_range", info.color_range))
    return flags


def _trim_filter(start_frame: int, end_frame: int | None) -> str | None:
    if start_frame <= 0 and end_frame is None:
        return None
    parts = []
    if start_frame > 0:
        parts.append(f"start_frame={start_frame}")
    if end_frame is not None:
        parts.append(f"end_frame={end_frame}")
    return "trim=" + ":".join(parts) + ",setpts=PTS-STARTPTS"


def _optional_float(value: object) -> float | None:
    if value in (None, "N/A", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(value: Fraction) -> str:
    return f"{float(value):.9f}"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _read_exact(stream, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == count:
                return None
            raise MediaError(
                f"ffmpeg closed the pipe mid-frame ({count - remaining} of {count} bytes)"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _drain_stderr(stream, sink: bytearray) -> None:
    data = stream.read()
    if data:
        sink.extend(data)


def _timecode_to_frames(
    hours: int, minutes: int, seconds: int, frames: int, fps: int, drop: bool
) -> int:
    if not drop:
        return ((hours * 60 + minutes) * 60 + seconds) * fps + frames
    drops = 2 * (fps // 30)
    total_minutes = hours * 60 + minutes
    return (
        ((hours * 3600) + (minutes * 60) + seconds) * fps
        + frames
        - drops * (total_minutes - total_minutes // 10)
    )


def _frames_to_timecode(frame_number: int, fps: int, drop: bool) -> tuple[int, int, int, int]:
    if frame_number < 0:
        raise MediaError(f"timecode frame index is negative ({frame_number})")
    if not drop:
        frame_number %= fps * 86400
        frames = frame_number % fps
        rest = frame_number // fps
        seconds = rest % 60
        rest //= 60
        minutes = rest % 60
        hours = (rest // 60) % 24
        return hours, minutes, seconds, frames
    drops = 2 * (fps // 30)
    frames_per_10min = fps * 600 - drops * 9
    frames_per_min = fps * 60 - drops
    frames_per_day = frames_per_10min * 6 * 24
    frame_number %= frames_per_day
    ten_min = frame_number // frames_per_10min
    remainder = frame_number % frames_per_10min
    if remainder < fps * 60:
        minute_in_block = 0
        frames_in = remainder
    else:
        remainder -= fps * 60
        minute_in_block = 1 + remainder // frames_per_min
        frames_in = (remainder % frames_per_min) + drops
    total_minutes = ten_min * 10 + minute_in_block
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    seconds = frames_in // fps
    frames = frames_in % fps
    return hours, minutes, seconds, frames
