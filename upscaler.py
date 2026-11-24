"""Upscale the native equirectangular strip to the 16:9 UHD raster.

The master scale is Lanczos-4 (``INTER_LANCZOS4``), about 195.7% on each axis:
7850×4416 becomes 15360×8640. The resize is wrapped horizontally so the
Lanczos kernel at the 0°/360° seam reads real pixels from the other side.
There is no unsharp mask and no neural model in this path. Both of those
rewrite sample values; Lanczos only interpolates them.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("hal250.upscale")

LANCZOS_SUPPORT = 4


class UpscaleError(RuntimeError):
    """The selected upscaler cannot process this frame."""


def resize_wrap_horizontal(
    image: np.ndarray,
    out_width: int,
    out_height: int,
    interpolation: int,
) -> np.ndarray:
    """Resize ``image`` to ``out_width`` × ``out_height`` with a wrapped X axis.

    OpenCV's resize kernel samples off the edge of the buffer. Padding a
    whole number of destination pixels worth of wrapped source, resizing,
    and cropping puts that kernel inside real pixels instead of a reflected
    border. The pad is chosen so the crop origin lands on an integer pixel.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise UpscaleError(f"expected HxWx3 image, got shape {image.shape}")
    src_h, src_w = image.shape[:2]
    if out_width < 2 or out_height < 2:
        raise UpscaleError(f"output size must be at least 2×2, got {out_width}×{out_height}")
    if src_w == out_width and src_h == out_height:
        return np.ascontiguousarray(image)

    scale = out_width / src_w
    support = LANCZOS_SUPPORT if interpolation == cv2.INTER_LANCZOS4 else 2
    min_pad = int(math.ceil(support * max(1.0, src_w / out_width))) + 2
    pad = _aligned_pad(src_w, out_width, min_pad)
    if pad >= src_w:
        # The alignment step can ask for a pad wider than the frame on tiny
        # previews. Repeat the row rather than indexing off the end.
        reps = int(math.ceil(pad / src_w))
        tiled = np.concatenate([image] * (reps * 2 + 1), axis=1)
        start = src_w * reps - pad
        wrapped = tiled[:, start : start + src_w + 2 * pad]
    else:
        wrapped = np.concatenate([image[:, -pad:], image, image[:, :pad]], axis=1)

    crop_x = pad * out_width // src_w
    full_w = out_width + 2 * crop_x
    resized = cv2.resize(wrapped, (full_w, out_height), interpolation=interpolation)
    if resized.dtype != image.dtype:
        resized = np.clip(np.rint(resized), 0, 65535).astype(image.dtype)
    crop = resized[:, crop_x : crop_x + out_width]
    if crop.shape[1] != out_width or crop.shape[0] != out_height:
        raise UpscaleError(
            f"wrapped resize produced {crop.shape[1]}×{crop.shape[0]}, "
            f"expected {out_width}×{out_height}"
        )
    return np.ascontiguousarray(crop)


@dataclass
class LanczosUpscaler:
    """Lanczos-4 resize to an exact size. No sharpening and no tone change."""

    def upscale(self, frame: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
        resized = resize_wrap_horizontal(frame, out_width, out_height, cv2.INTER_LANCZOS4)
        if resized.dtype != np.uint16:
            resized = np.clip(np.rint(resized), 0, 65535).astype(np.uint16)
        return resized


class AIUpscaler:
    """Tile-based xN super-resolution with a wrapped panorama seam.

    ``model`` is an ONNX file or a TorchScript module. The network is called
    with ``NCHW`` float32 RGB in ``[0, 1]`` and must return the same layout.
    After the network's integer scale, Lanczos fits the strip to the exact
    delivery size (the HAL 250 master is 195.7% rather than a pure 2×).
    """

    def __init__(
        self,
        model: Path | None = None,
        *,
        tile: int = 512,
        tile_pad: int = 32,
        modulo: int = 8,
        scale: int | None = None,
        device: str = "cpu",
        wrap_pad: int = 32,
        backend=None,
    ) -> None:
        if tile < 16:
            raise UpscaleError("--sr-tile must be at least 16")
        if tile_pad < 0 or wrap_pad < 0:
            raise UpscaleError("tile padding cannot be negative")
        if modulo < 1:
            raise UpscaleError("--sr-modulo must be >= 1")
        self.tile = tile
        self.tile_pad = tile_pad
        self.modulo = modulo
        self.wrap_pad = wrap_pad
        self.device = device
        if backend is not None:
            if scale is None or scale < 1:
                raise UpscaleError("an injected AI backend needs a positive scale")
            self._run_nchw = backend
            self.scale = scale
            self.backend_name = "injected"
        else:
            if model is None:
                raise UpscaleError("AI upscaler requires --sr-model")
            self._run_nchw, self.backend_name = _load_model(model, device)
            self.scale = scale if scale is not None else _detect_scale(self._run_nchw)
        if self.scale < 1:
            raise UpscaleError(f"super-resolution scale must be >= 1, got {self.scale}")
        log.info(
            "AI upscaler %s scale %dx on %s (tile %d, tile pad %d, seam pad %d)",
            self.backend_name,
            self.scale,
            device,
            self.tile,
            self.tile_pad,
            self.wrap_pad,
        )

    def upscale(self, frame: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
        if frame.dtype != np.uint16:
            raise UpscaleError(f"AI upscaler expects uint16 RGB, got {frame.dtype}")
        height, width = frame.shape[:2]
        wrap = min(self.wrap_pad, max(0, width // 4))
        if wrap:
            padded = np.concatenate([frame[:, -wrap:], frame, frame[:, :wrap]], axis=1)
        else:
            padded = frame
        scaled = self._upscale_tiled(padded)
        if wrap:
            crop = wrap * self.scale
            if scaled.shape[1] <= crop * 2:
                raise UpscaleError("super-resolution output is smaller than the seam padding")
            scaled = scaled[:, crop:-crop]
        expected_w = width * self.scale
        expected_h = height * self.scale
        if scaled.shape[0] < expected_h or scaled.shape[1] < expected_w:
            raise UpscaleError(
                f"super-resolution returned {scaled.shape[1]}×{scaled.shape[0]}, "
                f"expected at least {expected_w}×{expected_h}"
            )
        scaled = scaled[:expected_h, :expected_w]
        if scaled.shape[1] == out_width and scaled.shape[0] == out_height:
            return np.ascontiguousarray(scaled)
        return resize_wrap_horizontal(scaled, out_width, out_height, cv2.INTER_LANCZOS4)

    def _upscale_tiled(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = self.scale
        out = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
        ys = _tile_starts(height, self.tile)
        xs = _tile_starts(width, self.tile)
        image = frame.astype(np.float32) * (1.0 / 65535.0)
        for y in ys:
            for x in xs:
                tile_h = min(self.tile, height - y)
                tile_w = min(self.tile, width - x)
                y0 = max(0, y - self.tile_pad)
                x0 = max(0, x - self.tile_pad)
                y1 = min(height, y + tile_h + self.tile_pad)
                x1 = min(width, x + tile_w + self.tile_pad)
                patch = image[y0:y1, x0:x1]
                pred = self._run_patch(patch)
                top = (y - y0) * scale
                left = (x - x0) * scale
                block = pred[top : top + tile_h * scale, left : left + tile_w * scale]
                out[y * scale : (y + tile_h) * scale, x * scale : (x + tile_w) * scale] = block
        np.clip(out, 0.0, 1.0, out=out)
        return np.rint(out * 65535.0).astype(np.uint16)

    def _run_patch(self, patch: np.ndarray) -> np.ndarray:
        ph, pw = patch.shape[:2]
        pad_b = (self.modulo - ph % self.modulo) % self.modulo
        pad_r = (self.modulo - pw % self.modulo) % self.modulo
        if pad_b or pad_r:
            padded = cv2.copyMakeBorder(patch, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)
        else:
            padded = patch
        nchw = np.transpose(padded, (2, 0, 1))[None].astype(np.float32, copy=False)
        try:
            pred = self._run_nchw(nchw)
        except Exception as exc:  # noqa: BLE001 - surface model errors as upscaler failures
            raise UpscaleError(
                "super-resolution model rejected a tile "
                f"({nchw.shape[3]}×{nchw.shape[2]}). "
                "Check --sr-tile, --sr-modulo, and that the file is an NCHW RGB 0-1 network. "
                f"Backend said: {exc}"
            ) from exc
        pred = np.asarray(pred)
        if pred.ndim != 4 or pred.shape[0] != 1 or pred.shape[1] != 3:
            raise UpscaleError(
                f"model returned shape {tuple(pred.shape)}; expected NCHW with 3 channels"
            )
        hwc = np.transpose(pred[0], (1, 2, 0)).astype(np.float32, copy=False)
        return hwc[: ph * self.scale, : pw * self.scale]


def _aligned_pad(src_w: int, dst_w: int, min_pad: int) -> int:
    """Smallest pad ``>= min_pad`` such that ``pad * dst_w`` is divisible by ``src_w``."""
    step = src_w // math.gcd(src_w, dst_w)
    pad = step
    if pad < min_pad:
        pad = int(math.ceil(min_pad / step)) * step
    return pad


def _tile_starts(length: int, tile: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length, tile))
    if starts[-1] + tile > length:
        starts[-1] = length - tile
    # range() can leave a duplicate when length is an exact multiple plus the adjust.
    deduped: list[int] = []
    for start in starts:
        if not deduped or deduped[-1] != start:
            deduped.append(start)
    return deduped


def _detect_scale(run_nchw) -> int:
    last_error: Exception | None = None
    for size in (64, 32, 128):
        dummy = np.zeros((1, 3, size, size), dtype=np.float32)
        try:
            pred = np.asarray(run_nchw(dummy))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if pred.ndim != 4:
            raise UpscaleError(f"model returned shape {tuple(pred.shape)} on a probe tensor")
        scale_y = pred.shape[2] / size
        scale_x = pred.shape[3] / size
        if scale_y != scale_x or scale_y < 1 or not float(scale_y).is_integer():
            raise UpscaleError(
                f"model scale is {scale_x}×{scale_y}; pass an integer-scale x2/x4 network "
                "or set --sr-scale"
            )
        return int(scale_y)
    raise UpscaleError(
        "could not probe the super-resolution scale with a blank tensor. "
        f"Last error: {last_error}"
    )


def _load_model(path: Path, device: str):
    if not path.is_file():
        raise UpscaleError(f"super-resolution model does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return _load_onnx(path, device), f"onnx:{path.name}"
    if suffix in {".pt", ".pth", ".ts"}:
        return _load_torchscript(path, device), f"torchscript:{path.name}"
    raise UpscaleError(
        f"unsupported model type {suffix!r}. Export the network to ONNX or TorchScript "
        "(.onnx, .pt, .ts). A raw Real-ESRGAN state dict is not loaded directly."
    )


def _load_onnx(path: Path, device: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise UpscaleError(
            "AI upscaler needs onnxruntime for .onnx models. "
            "Install it with: pip install onnxruntime  (or onnxruntime-gpu)"
        ) from exc
    available = set(ort.get_available_providers())
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise UpscaleError(
                "onnxruntime has no CUDAExecutionProvider. Install onnxruntime-gpu, "
                "or run with --device cpu."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "auto":
        providers = [
            name
            for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if name in available
        ]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    input_name = session.get_inputs()[0].name

    def run_nchw(nchw: np.ndarray) -> np.ndarray:
        outputs = session.run(None, {input_name: np.ascontiguousarray(nchw)})
        return outputs[0]

    return run_nchw


def _load_torchscript(path: Path, device: str):
    try:
        import torch
    except ImportError as exc:
        raise UpscaleError(
            "AI upscaler needs PyTorch for TorchScript models. Install it with: pip install torch"
        ) from exc
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise UpscaleError("PyTorch CUDA is not available. Use --device cpu.")
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")
    try:
        module = torch.jit.load(str(path), map_location=torch_device)
    except Exception as exc:  # noqa: BLE001
        raise UpscaleError(
            f"could not load {path} as TorchScript ({exc}). "
            "Export with torch.jit.trace or torch.jit.script. "
            "A plain .pth state dict needs the matching network class and is not loaded here."
        ) from exc
    module.eval()

    def run_nchw(nchw: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(nchw)).to(torch_device)
            pred = module(tensor)
            return pred.detach().float().cpu().numpy()

    return run_nchw
