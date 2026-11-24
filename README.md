# skinthefish

Unwrap circular 250° fisheye plates from a RED KOMODO 6K with an Entaniya HAL 250 3.6 mm lens, mounted vertically and aimed at the zenith, into a 360° equirectangular ProRes master.

The delivery raster is 16:9, and the captured polar range fills the frame. There is no black bar and no letterbox.

| `--size` | Delivery | Unwrap, then Lanczos-4 |
| --- | --- | --- |
| `16k` (default) | 15360×8640, four times UHD | 7850×4416, then about 195.7% |
| `8k` | 7680×4320, two times UHD | 3925×2208, then the same 195.7% |

The picture is processed as `yuv444p16le`. That is a higher-precision copy of the source YUV codes, not a conversion to RGB. Nothing is graded, sharpened, or range-lifted. The only thing that changes a sample is geometric resampling.

## Optical model

The KOMODO active picture is 6144×3240 (27.03×14.26 mm, 4.4 μm). The HAL 250 is equidistant, `r = f · θ`. With the circle centered on the sensor:

| | |
| --- | --- |
| Optical center, full 6144×3240 frame | `(3072, 1620)` |
| Optical center, 3240×3240 circle window | `(1620, 1620)` |
| 250° circle radius | `1620 px` (polar angle 125°) |
| Horizon (90° polar, 180° across) | `1166.4 px` on a full-resolution plate, which is 72% of the way down the frame |

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
2. `cv2.resize` with `INTER_LANCZOS4` scales that strip onto the chosen delivery size.

`--size 16k` uses the numbers below. `--size 8k` halves every one of them, so the scale factor stays the same.

| Axis | From | To | Factor |
| --- | --- | --- | --- |
| Width | 7850 | 15360 | 1.9566879 (195.67%) |
| Height | 4416 | 8640 | 1.9565217 (195.65%) |

4416 is `round(7850 × 9/16)`, so the working frame is already 16:9. 8640 is exactly `15360 × 9/16`. The two factors differ only because both heights are integers. 8K is 3925×2208 resized to 7680×4320.

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

A full 15360×8640 `yuv444p16` frame is about 760 MB before encoder scratch. An 8K frame is a quarter of that, about 190 MB. Expect several gigabytes of RAM for 16K. Use `--size 8k` or `--preview-scale` for the first look.

## Commands

Calibrate the circle before a batch. This writes a PNG (green = 90° horizon, yellow = 250° edge, white crosshair) and a JSON sidecar. `F` is the direction that will sit in the center of the panorama.

```bash
python3 main.py /media/A001_C001.mov --calibrate
```

Nudge center and radius until the yellow ring sits on the image circle, then render. Positive yaw turns the view clockwise. With no `-o`, the output is `{name}_16k.mov` or `{name}_8k.mov`.

A 3240×3240 file is treated as the circle's bounding square. The default center is then `(1620, 1620)` and the radius is `1620`, which is the full-sensor circle with the side margins cropped away. `--dry-run` prints that assumption. The yellow ring on the calibration still is the check.

```bash
python3 main.py /media/A001_C001.mov \
  --center-x 3071.5 --center-y 1618.0 --radius 1616.0 \
  --yaw-offset 90 \
  --calibration examples/lens_calibration.example.json \
  --size 16k \
  -o /renders/A001_C001_16k.mov
```

Batch a folder. Every file becomes a 16:9 master in the output folder.

```bash
python3 main.py /media/day01/ \
  --output-dir /renders/day01 \
  --overwrite
```

Quarter-resolution look-dev of the 8K raster, still 16:9, still Lanczos-4:

```bash
python3 main.py clip.mov --size 8k --preview-scale 4 -o /tmp/lookdev.mov
```

## App

`studio.py` opens a page on this computer. It shows one frame of the fisheye and the unwrapped preview side by side. Export stays off until that preview is on screen. You then pick the ProRes codec and a save path.

The pictures in the page are a viewing image. The file Export writes is the full 8K or 16K master, still in the source YUV, with the same Lanczos path as the command line.

```bash
python3 studio.py
```

On Windows, double-click `Skinthefish.bat` in this folder. The first run installs the Python packages and, if needed, FFmpeg. Leave the black window open while the page is open.

Before that, install Python 3.12 from [python.org](https://www.python.org/downloads/windows/) and tick **Add python.exe to PATH**. If FFmpeg does not install from the script, run `winget install Gyan.FFmpeg`, close the window, and double-click the script again. If `winget` is missing, download the essentials build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add its `bin` folder to PATH.

What to check on a 3240×3240 clip:

1. Open the movie. The header should show 3240×3240, and the center should start at 1620, 1620 with radius 1620.
2. The yellow ring sits on the edge of the circle. If it does not, nudge Center X, Center Y, and Circle radius. The preview updates, and Export turns off until the new unwrap is visible.
3. Choose 8K or 16K. 8K is the practical first export. 16K needs several gigabytes free.
4. Pick the ProRes codec. "Same as the source" keeps the flavor of the file you opened. The other choices write Proxy, LT, 422, HQ, 4444, or 4444 XQ.
5. Export. The page shows frame progress. The movie lands at the path you chose.

The input has to already be Apple ProRes. H.264, a still sequence, or an R3D file is refused.

On macOS, `brew install ffmpeg python` and then `python3 studio.py`. There is no Windows-only path in the unwrap itself.

Useful flags: `--interp cubic` (the remap kernel only; the upscale stays Lanczos-4), `--start-frame`, `--end-frame`, `--dry-run`, `-v`. A half-res 3072×1620 proxy of the same framing scales the default center and radius automatically. An explicit center, radius, or calibration file is used as written.

## What is preserved

| Source | Output |
| --- | --- |
| ProRes profile fourcc (`apco` `apcs` `apcn` `apch` `ap4h` `ap4x`) | same profile, unless the app or `--profile` picks another |
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
- Width and height are stored as 16-bit fields (maximum 65535), and 4:2:2 needs an even width. 15360×8640 and 7680×4320 both divide by 16. On FFmpeg 6.1, `prores_ks` encoded 15360×7680 and a piped 15360×5333 HQ frame and probed them back unchanged.
- The input has to already be ProRes. The tool will not invent a profile for H.264 or a camera raw file.

## Layout

| File | Role |
| --- | --- |
| `main.py` | Command line, calibration overlay, render loop |
| `geometry.py` | Equidistant model, remap grids, 16:9 raster sizes |
| `upscaler.py` | Wrapped Lanczos-4 resize to the delivery size |
| `media_io.py` | ffprobe, `yuv444p16le` decode, `prores_ks` encode |
| `studio.py` | Preview page: original, unwrap, then ProRes export |
| `web/index.html` | The page the app serves |
| `Skinthefish.bat` | Windows double-click install and launch |
| `gui.py` | Older tkinter form that shells out to `main.py` |
| `examples/lens_calibration.example.json` | Starting calibration |

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

The geometry tests lock the 1166.4 px horizon, the 3240×3240 circle window, and both the 15360×8640 and 7680×4320 rasters. The pipeline tests encode a short ProRes fixture with ffmpeg and check profile, timecode, color tags, audio, that a horizon marker lands in the center of the panorama, and that a flat color comes back with the same YUV codes.
