# skinthefish

Unwrap circular 250° fisheye plates from a RED KOMODO 6K with an Entaniya HAL 250 3.6 mm lens, mounted vertically and aimed at the zenith, into a 360° equirectangular ProRes master.

The delivery raster is 15360×7680 (16K, 2:1 LatLong) or the active 15360×5333 strip with no nadir pad. The encode keeps the source Apple ProRes profile, frame rate, timecode, audio, and color tags. Sampling stays in 16-bit RGB (`rgb48le`) from decode through the remap.

## Optical model

The KOMODO active picture is 6144×3240 (27.03×14.26 mm, 4.4 μm). The HAL 250 is equidistant, `r = f · θ`. With the circle centered on the sensor:

| | |
| --- | --- |
| Optical center | `(3072, 1620)` |
| 250° circle radius | `1620 px` (polar angle 125°) |
| Horizon (90° polar, 180° across) | `1166.4 px`, which is 72% of the way down the active strip |

Azimuth `φ` runs across the panorama from `−π` at the left edge to just under `+π` at the right edge. Polar angle `θ` runs from 0° (zenith) at the top of the strip to 125° at the bottom. Positive `--yaw-offset` rotates the view clockwise in the sensor image, so the content of the panorama moves left. Yaw 0 puts the top of the sensor (the direction above the optical center) in the middle of the frame.

Distortion coefficients bend that curve and are renormalized so the calibrated radius still sits on `θ_max`:

```
θ̂ = θ / θ_max
f(θ̂) = θ̂ · (1 + k1 θ̂² + k2 θ̂⁴ + k3 θ̂⁶)
r(θ) = R · f(θ̂) / f(1)
```

With `k1 = k2 = k3 = 0` this is `r(θ) = R_250 · (θ / 125°)`.

The unwrap is built at 7850×2726, then scaled by 15360/7850 (195.67%) to 15360×5333. `full-latlong` appends 2347 rows of black (125° is 5333/7680 of a 180° LatLong) to make 15360×7680. `active-crop` writes 15360×5333 for a custom UV dome or cylinder.

## Install

FFmpeg 6 or newer, with the `prores_ks` encoder (the Ubuntu and Debian `ffmpeg` packages have it):

```bash
sudo apt-get install ffmpeg
python -m pip install -r requirements.txt
```

Check the toolchain:

```bash
ffmpeg -hide_banner -encoders | grep prores_ks
python main.py --help
```

Optional, only for `--upscaler ai`:

```bash
python -m pip install onnxruntime          # .onnx on CPU
python -m pip install onnxruntime-gpu      # .onnx on NVIDIA
python -m pip install torch                # TorchScript .pt / .ts
```

The GUI (`python gui.py`) needs a display and `tkinter` (`sudo apt-get install python3-tk` on Debian/Ubuntu).

A full 16K frame is about 700 MB of `rgb48` before encoder scratch. Expect several gigabytes of RAM per frame. Use `--preview-scale` for look-dev.

## Commands

Calibrate the circle before a batch. This writes a PNG (green = 90° horizon, yellow = 250° edge, white crosshair) and a JSON sidecar. `F` is the direction that will sit in the center of the panorama.

```bash
python main.py /media/A001_C001.mov --calibrate
```

Nudge center and radius until the yellow ring sits on the image circle, then render. Positive yaw turns the view clockwise.

```bash
python main.py /media/A001_C001.mov \
  --center-x 3071.5 --center-y 1618.0 --radius 1616.0 \
  --yaw-offset 90 \
  --calibration examples/lens_calibration.example.json \
  --output-mode full-latlong \
  --upscaler lanczos \
  -o /renders/A001_C001_latlong.mov
```

Batch a folder to the active 15360×5333 crop:

```bash
python main.py /media/day01/ \
  --output-dir /renders/day01 \
  --output-mode active-crop \
  --overwrite
```

Quarter-resolution look-dev, then an ONNX x2 pass. The network must be NCHW float RGB in `[0, 1]` (Real-ESRGAN / SwinIR style). Weights are not bundled. The 0°/360° seam is padded with a horizontal wrap before tiling, then the x2 result is fitted to the exact delivery size.

```bash
python main.py clip.mov --preview-scale 4 -o /tmp/lookdev.mov

python main.py clip.mov \
  --upscaler ai \
  --sr-model models/realesrgan_x2.onnx \
  --device cuda \
  --output-mode full-latlong \
  -o clip_16k.mov
```

Useful flags: `--interp cubic`, `--no-unsharp`, `--start-frame`, `--end-frame`, `--black-level`, `--dry-run`, `-v`. A half-res 3072×1620 proxy of the same framing scales the default center and radius automatically. An explicit center, radius, or calibration file is used as written.

`--black-level` is a 16-bit code in the full-range RGB image that ffmpeg is given. `0` re-encodes as black. The latlong pad and the samples outside the circle both use it.

## What is preserved

| Source | Output |
| --- | --- |
| ProRes profile fourcc (`apco` `apcs` `apcn` `apch` `ap4h` `ap4x`) | same `-profile:v` on `prores_ks` |
| 422 profiles | `yuv422p10le` |
| 4444 and 4444 XQ | `yuv444p10le` into the encoder |
| Frame rate | exact rational (`24000/1001` stays that) |
| Timecode | shifted when `--start-frame` is not 0. Drop-frame (`;`) is kept for 29.97 and 59.94 |
| Audio | stream-copied |
| Color primaries, transfer, matrix | passed through when ffprobe reports them |
| Global metadata such as a reel comment | copied |

Vendor id on the ProRes frame header is `apl0`.

Limits, so they are not a surprise on a render node:

- `prores_ks` accepts 10-bit pixel formats. 4444 files probe back as `yuv444p12le` because the codec codeword is 12 bits wide; the extra two bits are not stored. Decode and the remap still run in `rgb48`, so nothing is quantized to 8-bit on the way through.
- Alpha on 4444 is dropped. The picture is RGB.
- Width and height are stored as 16-bit fields (maximum 65535), and 4:2:2 needs an even width. On FFmpeg 6.1 `prores_ks`, both delivery sizes were encoded and probed back unchanged: 15360×7680, and a piped 15360×5333 HQ frame (`apch`). 5333 is not a multiple of 16; the encoder pads slices internally and still stores 5333.
- The input has to already be ProRes. The tool will not invent a profile for H.264 or a camera raw file.

## Layout

| File | Role |
| --- | --- |
| `main.py` | Command line, calibration overlay, render loop |
| `geometry.py` | Equidistant model, remap grids, raster sizes |
| `upscaler.py` | Wrapped Lanczos-4 + unsharp, optional tiled AI |
| `media_io.py` | ffprobe, `rgb48le` decode, `prores_ks` encode |
| `gui.py` | Optional tkinter form that shells out to `main.py` |
| `examples/lens_calibration.example.json` | Starting calibration |

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The geometry tests lock the 1166.4 px horizon, the 7850×2726 and 15360×5333 rasters, and the yaw/seam orientation. The pipeline tests encode a short ProRes fixture with ffmpeg and check profile, timecode, color tags, audio, and that a horizon marker in front of the lens lands in the center of the panorama.
