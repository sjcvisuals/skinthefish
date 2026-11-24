"""Local preview app for the HAL 250 unwrap.

Opens a page on this computer that shows one frame of the fisheye, the
unwrapped 16:9 preview, and an Export button. Export stays unavailable until
that preview has been drawn. The master file is still the YUV Lanczos render
from ``main.py``. The pictures in the page are a viewing conversion.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import shutil
import threading
import traceback
import urllib.parse
import webbrowser

import cv2
import numpy as np

from geometry import GeometryError, LensModel, autoscale_default_lens, delivery_raster
from main import (
    INTERPOLATION,
    draw_calibration_overlay,
    render_movie,
    resolve_lens,
    unwrap_and_upscale,
)
from media_io import (
    OUTPUT_PROFILE_CHOICES,
    MediaError,
    MediaInfo,
    probe_media,
    require_prores,
)
from upscaler import LanczosUpscaler

log = logging.getLogger("hal250.studio")

WEB_DIR = Path(__file__).resolve().parent / "web"
SOURCE_VIEW_WIDTH = 720
_PREVIEW_LOCK = threading.Lock()


class StudioError(RuntimeError):
    """The page sent a path or a setting the app cannot use."""


class Studio:
    """One open clip, one preview cache, and at most one export."""

    def __init__(self) -> None:
        self._frame_key: tuple[str, int] | None = None
        self._frame: np.ndarray | None = None
        self._info: MediaInfo | None = None
        self.job: dict[str, object] = {
            "state": "idle",
            "frame": 0,
            "total": None,
            "message": "",
            "error": "",
            "output": "",
        }
        self._job_thread: threading.Thread | None = None

    def health(self, initial_path: Path | None = None) -> dict[str, object]:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        return {
            "ffmpeg": bool(ffmpeg and ffprobe),
            "message": ""
            if ffmpeg and ffprobe
            else "FFmpeg is not on PATH. On Windows, run Skinthefish.bat again after installing it.",
            "initial_path": str(initial_path) if initial_path else "",
            "profiles": list(OUTPUT_PROFILE_CHOICES),
        }

    def open_clip(self, path: Path) -> dict[str, object]:
        info = self._probe(path)
        lens = autoscale_default_lens(LensModel(), info.width, info.height)
        return {
            "path": str(info.path),
            "width": info.width,
            "height": info.height,
            "fps": f"{info.fps.numerator}/{info.fps.denominator}",
            "frame_count": info.frame_count or 1,
            "profile": info.profile,
            "profile_label": info.profile_label,
            "timecode": info.timecode or "",
            "center_x": lens.center_x,
            "center_y": lens.center_y,
            "radius": lens.radius,
            "yaw_offset": lens.yaw_offset_deg,
            "square": info.width == info.height,
        }

    def source_jpeg(self, params: dict[str, str]) -> bytes:
        info = self._info_for(params)
        frame_index = _frame_index(params, info)
        frame = self._cached_frame(info, frame_index)
        lens = _lens_from_params(params, info)
        view, view_lens = scale_frame_and_lens(frame, lens, SOURCE_VIEW_WIDTH)
        if params.get("rings", "1") != "0":
            image = draw_calibration_overlay(view, view_lens)
        else:
            image = cv2.cvtColor((view >> 8).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return _jpeg(image)

    def unwrap_jpeg(self, params: dict[str, str]) -> bytes:
        info = self._info_for(params)
        frame_index = _frame_index(params, info)
        frame = self._cached_frame(info, frame_index)
        lens = _lens_from_params(params, info)
        size = params.get("size") or "8k"
        raster = delivery_raster(lens.theta_max_deg, _display_scale(size), size)
        maps = build_maps(raster.native_width, raster.native_height, lens)
        picture = unwrap_and_upscale(frame, maps, LanczosUpscaler(), raster, INTERPOLATION["lanczos4"])
        view = cv2.cvtColor((picture >> 8).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return _jpeg(view)

    def start_export(self, body: dict[str, object]) -> dict[str, object]:
        path = Path(str(body.get("path") or ""))
        output = Path(str(body.get("output") or ""))
        if not str(body.get("output") or "").strip():
            raise StudioError("Choose a file name for the export.")
        info = self._probe(path)
        job = {
            "state": "running",
            "frame": 0,
            "total": info.frame_count,
            "message": "starting",
            "error": "",
            "output": str(output),
        }
        args = _export_args(body, job)
        with _PREVIEW_LOCK:
            if self._job_thread is not None and self._job_thread.is_alive():
                raise StudioError("An export is already running.")
            self.job = job
            thread = threading.Thread(
                target=self._run_export,
                args=(args, info.path, output),
                daemon=True,
            )
            self._job_thread = thread
            thread.start()
        return {"state": "running", "output": str(output)}

    def job_status(self) -> dict[str, object]:
        with _PREVIEW_LOCK:
            return dict(self.job)

    def _run_export(self, args: argparse.Namespace, source: Path, output: Path) -> None:
        try:
            render_movie(args, source, output)
        except Exception as exc:  # noqa: BLE001 - the page has to show the failure
            log.error("export failed\n%s", traceback.format_exc())
            with _PREVIEW_LOCK:
                self.job["state"] = "error"
                self.job["error"] = str(exc)
            return
        with _PREVIEW_LOCK:
            self.job["state"] = "done"
            self.job["message"] = f"wrote {output}"

    def _probe(self, path: Path) -> MediaInfo:
        if not str(path):
            raise StudioError("Choose a ProRes movie.")
        if not path.is_file():
            raise StudioError(f"file not found: {path}")
        try:
            info = probe_media(path)
            require_prores(info)
        except MediaError as exc:
            raise StudioError(str(exc)) from exc
        self._info = info
        return info

    def _info_for(self, params: dict[str, str]) -> MediaInfo:
        path = Path(params.get("path") or "")
        if self._info is not None and self._info.path == path:
            return self._info
        return self._probe(path)

    def _cached_frame(self, info: MediaInfo, frame_index: int) -> np.ndarray:
        key = (str(info.path), frame_index)
        with _PREVIEW_LOCK:
            if self._frame_key == key and self._frame is not None:
                return self._frame
        frame = read_rgb_frame(info, frame_index)
        with _PREVIEW_LOCK:
            self._frame_key = key
            self._frame = frame
        return frame


def scale_frame_and_lens(frame: np.ndarray, lens: LensModel, width: int):
    """Shrink a viewing frame and the circle together so the overlay stays readable."""
    height, current = frame.shape[:2]
    if current <= width:
        return frame, lens
    scale = width / current
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized, replace(
        lens,
        center_x=lens.center_x * scale,
        center_y=lens.center_y * scale,
        radius=lens.radius * scale,
    )


def build_maps(width: int, height: int, lens):
    from geometry import build_remap_maps

    return build_remap_maps(width, height, lens)


def read_rgb_frame(info: MediaInfo, frame_index: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Read one RGB viewing frame. ProRes is intra-frame, so a timestamp seek is exact."""
    from fractions import Fraction

    if frame_index < 0:
        raise StudioError("frame must be >= 0")
    if info.frame_count is not None and frame_index >= info.frame_count:
        raise StudioError(f"frame {frame_index} is past the end of a {info.frame_count}-frame clip")
    timestamp = float(Fraction(frame_index, 1) / info.fps)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(info.path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-",
    ]
    import subprocess

    try:
        raw = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as exc:
        raise StudioError(f"could not read frame {frame_index}") from exc
    expected = info.width * info.height * 6
    if len(raw) < expected:
        raise StudioError(f"frame {frame_index} was shorter than {info.width}×{info.height}")
    return np.frombuffer(raw[:expected], dtype="<u2").reshape(info.height, info.width, 3).copy()


def browse(kind: str, suggested: str = "") -> str:
    """Open the operating-system file dialog. The page can also take a typed path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # noqa: BLE001 - tk is optional on some Python installs
        raise StudioError(
            "This Python has no file dialog. Paste the full path into the box. "
            "The python.org Windows installer includes the dialog."
        ) from exc
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    initial = suggested.strip()
    if kind == "save":
        chosen = filedialog.asksaveasfilename(
            title="Export ProRes",
            defaultextension=".mov",
            filetypes=[("QuickTime movie", "*.mov")],
            initialfile=Path(initial).name if initial else "unwrap.mov",
        )
    else:
        chosen = filedialog.askopenfilename(
            title="Open a ProRes movie",
            filetypes=[("Movies", "*.mov *.mxf *.mp4 *.qt"), ("All files", "*.*")],
        )
    root.destroy()
    if not chosen:
        raise StudioError("no file selected")
    return chosen


def _display_scale(size: str) -> int:
    """Preview scale that lands the on-screen unwrap near 960 pixels wide."""
    if size == "8k":
        return 8
    if size == "16k":
        return 16
    raise StudioError("size must be 8k or 16k")


def _frame_index(params: dict[str, str], info: MediaInfo) -> int:
    raw = params.get("frame") or "0"
    try:
        index = int(raw)
    except ValueError as exc:
        raise StudioError(f"frame must be a whole number, got {raw!r}") from exc
    last = (info.frame_count or 1) - 1
    return max(0, min(index, last))


def _lens_from_params(params: dict[str, str], info: MediaInfo):
    args = argparse.Namespace(
        calibration=None,
        center_x=_optional_float(params.get("center_x")),
        center_y=_optional_float(params.get("center_y")),
        radius=_optional_float(params.get("radius")),
        theta_max=None,
        yaw_offset=_optional_float(params.get("yaw")),
        k1=None,
        k2=None,
        k3=None,
    )
    try:
        return resolve_lens(args, info.width, info.height)
    except GeometryError as exc:
        raise StudioError(str(exc)) from exc


def _export_args(body: dict[str, object], job: dict[str, object]) -> argparse.Namespace:
    def number(name: str) -> float | None:
        value = body.get(name)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise StudioError(f"{name} must be a number") from exc

    def on_progress(frame: int, total: int | None, message: str) -> None:
        with _PREVIEW_LOCK:
            job["frame"] = frame
            job["total"] = total
            job["message"] = message

    return argparse.Namespace(
        calibration=None,
        center_x=number("center_x"),
        center_y=number("center_y"),
        radius=number("radius"),
        theta_max=None,
        yaw_offset=number("yaw"),
        k1=None,
        k2=None,
        k3=None,
        preview_scale=1,
        size=str(body.get("size") or "8k"),
        interp="lanczos4",
        start_frame=0,
        end_frame=None,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        overwrite=True,
        dry_run=False,
        profile=str(body.get("profile") or "match"),
        on_progress=on_progress,
    )


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise StudioError(f"expected a number, got {value!r}") from exc


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise StudioError("could not encode the preview")
    return encoded.tobytes()


def make_handler(studio: Studio, initial_path: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", (WEB_DIR / "index.html").read_bytes())
                return
            if parsed.path == "/api/health":
                self._json(studio.health(initial_path))
                return
            if parsed.path == "/api/job":
                self._json(studio.job_status())
                return
            params = {key: values[0] for key, values in urllib.parse.parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/api/source.jpg":
                    self._send(200, "image/jpeg", studio.source_jpeg(params))
                    return
                if parsed.path == "/api/unwrap.jpg":
                    self._send(200, "image/jpeg", studio.unwrap_jpeg(params))
                    return
            except StudioError as exc:
                self._json({"error": str(exc)}, status=400)
                return
            self._json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json({"error": "the request was not valid JSON"}, status=400)
                return
            try:
                if parsed.path == "/api/open":
                    self._json(studio.open_clip(Path(str(body.get("path") or ""))))
                    return
                if parsed.path == "/api/export":
                    self._json(studio.start_export(body))
                    return
                if parsed.path == "/api/browse":
                    self._json({"path": browse(str(body.get("kind") or "open"), str(body.get("suggested") or ""))})
                    return
            except StudioError as exc:
                self._json({"error": str(exc)}, status=400)
                return
            self._json({"error": "not found"}, status=404)

        def log_message(self, fmt: str, *args) -> None:
            log.debug("%s - " + fmt, self.address_string(), *args)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(
    initial_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve the page on localhost and open it in the browser."""
    from main import configure_logging

    configure_logging(False)
    studio = Studio()
    handler = make_handler(studio, initial_path)
    httpd = None
    last_error: OSError | None = None
    for candidate in range(port, port + 10):
        try:
            httpd = ThreadingHTTPServer((host, candidate), handler)
            port = candidate
            break
        except OSError as exc:
            last_error = exc
    if httpd is None:
        raise StudioError(f"could not listen on {host}:{port} ({last_error})")
    url = f"http://{host}:{port}/"
    log.info("Skinthefish is open at %s", url)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
