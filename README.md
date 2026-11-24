# skinthefish

Unwrap circular 250° fisheye plates from a RED KOMODO 6K with an Entaniya HAL 250 3.6 mm lens, mounted vertically and aimed at the zenith, into a 360° equirectangular ProRes master.

The delivery raster is always 15360×8640. That is exactly 16:9, and it is four times UHD (3840×2160). The captured polar range fills the frame. There is no black bar and no letterbox.

The picture is processed as `yuv444p16le`. That is a higher-precision copy of the source YUV codes, not a conversion to RGB. Nothing is graded, sharpened, or range-lifted. The only thing that changes a sample is geometric resampling.

## Optical model

The KOMODO active picture is 6144×3240 (27.03×14.26 mm, 4.4 μm). The HAL 250 is equidistant, `r = f · θ`. With the circle centered on the sensor:

| | |
| --- | --- |
| Optical center | `(3072, 1620)` |
| 250° circle radius | `1620 px` (polar angle 125°) |
| Horizon (90° polar, 180° across) | `1166.4 px`, which is 72% of the way down the frame |

Azimuth `φ` runs across the panorama from `−π` at the left edge to just under `+π` at the right edge. Polar angle `θ` runs from 0° (zenith) at the top of the frame to 125° at the bottom. Positive `--yaw-offset` rotates the view clockwise in the sensor image, so the content of the panorama moves left. Yaw 0 puts the top of the sensor (the direction above the optical center) in the middle of the frame.

Distortion coefficients bend that curve and are renormalized so the calibrated radius still sits on `θ_max`:

```
θ̂ = θ / θ_max
f(θ̂) = θ̂ · (1 + k1 θ̂² + k2 θ̂⁴ + k3 θ̂⁶)
r(θ) = R · f(θ̂) / f(1)
```

With `k1 = k2 = k3 = 0` this is `r(θ) = R_250 · (θ / 125°)`.

The 125° capture is stretched to fill 16:9. A square-pixel 2:1 latlong would leave the bottom of the frame empty, and this tool does not do that. Horizontal degrees-per-pixel and vertical degrees-per-pixel are therefore not equal. A VR player that assumes a 2:1 square-pixel latlong will show this frame stretched. The horizon stays at 72% of the height.

## How the upscale works

Two Lanczos-4 resamples, and nothing else.

1. `cv2.remap` builds the equirectangular picture at 7850×4416. The kernel is Lanczos-4 (`--interp lanczos4`, the default) or cubic (`--interp cubic`). This step samples the fisheye. It is not the upscale.
2. `cv2.resize` with `INTER_LANCZOS4` scales that strip onto 15360×8640.

The scale is the same on both axes, about 195.7%:

| Axis | From | To | Factor |
| --- | --- | --- | --- |
| Width | 7850 | 15360 | 1.9566879 (195.67%) |
| Height | 4416 | 8640 | 1.9565217 (195.65%) |

4416 is `round(7850 × 9/16)`, so the working frame is already 16:9. 8640 is exactly `15360 × 9/16`. The two factors differ only because both heights are integers.

Before the resize, the strip is padded by copying the opposite horizontal edge, using a pad width that lands the crop on a whole pixel. The Lanczos kernel at the 0°/360° seam therefore reads real neighbors, and the pad is cropped off afterward. The remap itself does not wrap vertically: the zenith has no opposite edge to copy.

There is no unsharp mask and no neural upscaler. Both of those rewrite texture and code values. Lanczos only interpolates. A flat color is unchanged. Where the picture has detail, neighboring samples are blended by the kernel. That blend is the resample.

Samples that fall outside the source picture are filled with code 0. That is only the missing texels. Valid pixels are not lifted or crushed.

## Color

Decode, remap, and upscale stay in Y, U, and V. The working format is `yuv444p16le`. 10-bit 4:2:2 sources are upsampled to 4:4:4 and stored in 16-bit codes, then encoded back to the source 4:2:2 or 4:4:4 pixel format. A measured flat color round-trips with a YUV delta of zero when the primaries, transfer, and matrix tags are copied onto both the raw pipe and the output.

The master is never converted to RGB. An RGB conversion applies a matrix and shifts the codes. The calibration PNG is the exception: it is a one-frame preview for drawing the rings, and it is not encoded back into the master.

Color primaries, transfer, matrix, and range are copied when ffprobe reports them. They are not invented, and there is no `colorspace` or `zscale` filter.

## Install

The same code runs on macOS and Linux. It uses little-endian `uint16` pipes, `pathlib`, and `ffmpeg` on `PATH`. There is no Mac in this test setup, so the ProRes checks below were run on Linux FFmpeg 6.1.

macOS, with [Homebrew](https://brew.sh):

```bash
brew install ffmpeg python
python3 -m pip install -r requirements.txt
ffmpeg -hide_banner -encoders | grep prores_ks
python3 main.py --help
```

The GUI needs Tk. The python.org macOS installer includes it. A Homebrew Python often needs `brew install python-tk`.

Linux:

```bash
sudo apt-get install ffmpeg
python3 -m pip install -r requirements.txt
```

On Debian and Ubuntu the `ffmpeg` package includes `prores_ks`. The GUI needs a display and `python3-tk`.

A full 15360×8640 `yuv444p16` frame is about 760 MB before encoder scratch. Expect several gigabytes of RAM per frame. Use `--preview-scale` for look-dev.

## Commands

Calibrate the circle before a batch. This writes a PNG (green = 90° horizon, yellow = 250° edge, white crosshair) and a JSON sidecar. `F` is the direction that will sit in the center of the panorama.

```bash
python3 main.py /media/A001_C001.mov --calibrate
```

Nudge center and radius until the yellow ring sits on the image circle, then render. Positive yaw turns the view clockwise. The output is `{name}_uhd.mov` when you do not pass `-o`.

```bash
python3 main.py /media/A001_C001.mov \
  --center-x 3071.5 --center-y 1618.0 --radius 1616.0 \
  --yaw-offset 90 \
  --calibration examples/lens_calibration.example.json \
  -o /renders/A001_C001_uhd.mov
```

Batch a folder. Every file becomes a 16:9 master in the output folder.

```bash
python3 main.py /media/day01/ \
  --output-dir /renders/day01 \
  --overwrite
```

Quarter-resolution look-dev, still 16:9, still Lanczos-4:

```bash
python3 main.py clip.mov --preview-scale 4 -o /tmp/lookdev.mov
```

Useful flags: `--interp cubic` (the remap kernel only; the upscale stays Lanczos-4), `--start-frame`, `--end-frame`, `--dry-run`, `-v`. A half-res 3072×1620 proxy of the same framing scales the default center and radius automatically. An explicit center, radius, or calibration file is used as written.

## What is preserved

| Source | Output |
| --- | --- |
| ProRes profile fourcc (`apco` `apcs` `apcn` `apch` `ap4h` `ap4x`) | same `-profile:v` on `prores_ks` |
| 422 profiles | `yuv422p10le` |
| 4444 and 4444 XQ | `yuv444p10le` into the encoder |
| Frame rate | exact rational (`24000/1001` stays that) |
| Timecode | shifted when `--start-frame` is not 0. Drop-frame (`;`) is kept for 29.97 and 59.94 |
| Audio | stream-copied |
| Color primaries, transfer, matrix, range | copied onto the raw YUV pipe and the output when ffprobe reports them |
| Global metadata such as a reel comment | copied |

Vendor id on the ProRes frame header is `apl0`.

Limits, so they are not a surprise on a render node:

- `prores_ks` accepts 10-bit pixel formats. 4444 files probe back as `yuv444p12le` because the codec codeword is 12 bits wide; the extra two bits are not stored. The remap still runs in 16-bit YUV, so nothing is quantized to 8-bit on the way through.
- Alpha on 4444 is dropped. The picture stays YUV.
- Width and height are stored as 16-bit fields (maximum 65535), and 4:2:2 needs an even width. 15360×8640 divides by 16 on both sides. On FFmpeg 6.1, `prores_ks` encoded 15360×7680 and a piped 15360×5333 HQ frame and probed them back unchanged. 8640 is aligned the same way.
- The input has to already be ProRes. The tool will not invent a profile for H.264 or a camera raw file.

## Layout

| File | Role |
| --- | --- |
| `main.py` | Command line, calibration overlay, render loop |
| `geometry.py` | Equidistant model, remap grids, 16:9 raster sizes |
| `upscaler.py` | Wrapped Lanczos-4 resize to the delivery size |
| `media_io.py` | ffprobe, `yuv444p16le` decode, `prores_ks` encode |
| `gui.py` | Optional tkinter form that shells out to `main.py` |
| `examples/lens_calibration.example.json` | Starting calibration |

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

The geometry tests lock the 1166.4 px horizon and the 7850×4416 to 15360×8640 rasters. The pipeline tests encode a short ProRes fixture with ffmpeg and check profile, timecode, color tags, audio, that a horizon marker lands in the center of the panorama, and that a flat color comes back with the same YUV codes.
