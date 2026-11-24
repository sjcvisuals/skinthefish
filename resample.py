"""Resample one fisheye frame onto the delivery grid.

The CPU path unwraps with Lanczos-4 at the native 16:9 size, then Lanczos-4
scales that strip to the delivery size. When PyTorch can see an NVIDIA GPU,
a master frame instead takes one bicubic sample on that GPU. Bicubic is still
interpolation: a flat color stays a flat color. It is not a neural upscaler.
ProRes encoding stays on the CPU either way, because NVIDIA cards do not
encode ProRes.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

import cv2
import numpy as np

from geometry import LensModel, RasterSpec, build_remap_maps
from upscaler import LanczosUpscaler, UpscaleError

log = logging.getLogger("hal250.resample")

# Below this many delivery pixels the CPU is already faster than a GPU copy.
_GPU_PIXEL_THRESHOLD = 2_000_000


def gpu_status() -> dict[str, object]:
    """Report whether this process can resample on an NVIDIA GPU."""
    name = _nvidia_name()
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        return {"available": True, "name": torch.cuda.get_device_name(0), "hint": ""}
    hint = ""
    if name:
        hint = (
            f"{name} is in this PC, but the unwrap is still on the CPU. "
            "Close this window and double-click Skinthefish-GPU.bat once. "
            "That installs the CUDA build of PyTorch so the graphics card does the resample."
        )
    return {"available": False, "name": name, "hint": hint}


def resolve_device(choice: str, pixels: int) -> str:
    """Pick ``cpu`` or ``cuda``. ``auto`` uses the GPU only for a real master."""
    selected = (choice or "auto").lower()
    if selected not in {"auto", "cpu", "cuda"}:
        raise UpscaleError(f"device must be auto, cpu, or cuda, got {choice!r}")
    if selected == "cpu":
        return "cpu"
    if gpu_status()["available"] and (selected == "cuda" or pixels >= _GPU_PIXEL_THRESHOLD):
        return "cuda"
    if selected == "cuda":
        raise UpscaleError(
            "CUDA was requested, but PyTorch cannot see a GPU. "
            "Double-click Skinthefish-GPU.bat and leave the NVIDIA driver installed."
        )
    return "cpu"


def cpu_unwrap(
    frame: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray],
    upscaler: LanczosUpscaler,
    raster: RasterSpec,
    interpolation: int,
) -> np.ndarray:
    """Remap at the native 16:9 size, then Lanczos-4 upscale to the delivery size."""
    native = cv2.remap(
        frame,
        maps[0],
        maps[1],
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if native.dtype != np.uint16:
        native = np.clip(np.rint(native), 0, 65535).astype(np.uint16)
    return upscaler.upscale(native, raster.delivery_width, raster.delivery_height)


class FrameResampler:
    """Hold the remap grid and turn one source frame into one delivery frame."""

    def __init__(self, lens: LensModel, raster: RasterSpec, interpolation: int, device: str) -> None:
        self.device = device
        self.interpolation = interpolation
        self.raster = raster
        self.lens = lens
        self._cuda = None
        self.maps = None
        self.upscaler = None
        if device == "cuda":
            map_x, map_y = build_remap_maps(raster.delivery_width, raster.delivery_height, lens)
            self._cuda = _CudaGrid(map_x, map_y)
            self.label = (
                f"{self._cuda.name} bicubic, one sample onto "
                f"{raster.delivery_width}×{raster.delivery_height}"
            )
            return
        self._use_cpu()
        kernel = "Lanczos-4" if interpolation == cv2.INTER_LANCZOS4 else "cubic"
        self.label = (
            f"CPU {kernel} {raster.native_width}×{raster.native_height} "
            f"then Lanczos-4 to {raster.delivery_width}×{raster.delivery_height}"
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self._cuda is not None:
            try:
                return self._cuda.apply(frame)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                log.warning("GPU ran out of memory (%s); falling back to the CPU", exc)
                self._cuda.release()
                self._cuda = None
                self.device = "cpu"
                self._use_cpu()
        return cpu_unwrap(frame, self.maps, self.upscaler, self.raster, self.interpolation)

    def _use_cpu(self) -> None:
        self.maps = build_remap_maps(self.raster.native_width, self.raster.native_height, self.lens)
        self.upscaler = LanczosUpscaler()


class _CudaGrid:
    def __init__(self, map_x: np.ndarray, map_y: np.ndarray) -> None:
        torch = _import_torch()
        if torch is None:
            raise UpscaleError("CUDA resample needs PyTorch. Double-click Skinthefish-GPU.bat.")
        self.torch = torch
        self._map_x = map_x
        self._map_y = map_y
        self.grid = None
        self.src: tuple[int, int] | None = None
        self.name = torch.cuda.get_device_name(0)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        torch = self.torch
        if frame.dtype != np.uint16 or frame.ndim != 3 or frame.shape[2] != 3:
            raise UpscaleError(
                f"GPU resample expected uint16 HxWx3, got {frame.dtype} {getattr(frame, 'shape', None)}"
            )
        height, width = frame.shape[:2]
        if self.grid is None or self.src != (width, height):
            norm = normalized_grid(self._map_x, self._map_y, width, height)
            self.grid = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).cuda()
            self.src = (width, height)
            self._map_x = None
            self._map_y = None
        image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device="cuda", dtype=torch.float32)
        sampled = torch.nn.functional.grid_sample(
            image,
            self.grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.clamp(0, 65535).round().to(torch.uint16)
        return sampled.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()

    def release(self) -> None:
        self.grid = None
        self.torch.cuda.empty_cache()


def normalized_grid(map_x: np.ndarray, map_y: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """OpenCV pixel coordinates to ``grid_sample(..., align_corners=False)``.

    That sampler puts pixel ``i`` at ``(i + 0.5) / size * 2 - 1``. An OpenCV
    map value of ``i`` already means that pixel center.
    """
    grid_x = (map_x + np.float32(0.5)) / np.float32(src_w) * np.float32(2.0) - np.float32(1.0)
    grid_y = (map_y + np.float32(0.5)) / np.float32(src_h) * np.float32(2.0) - np.float32(1.0)
    return np.stack((grid_x, grid_y), axis=-1).astype(np.float32, copy=False)


def _import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _nvidia_name() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else ""
