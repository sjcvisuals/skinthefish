"""The preview app: open a clip, show both pictures, and only then export."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from main import main
from media_io import MediaError, probe_media, resolve_output_profile
from studio import Studio, make_handler
from tests.test_pipeline import _encode_prores


def test_profile_choice_can_leave_the_source_codec():
    assert resolve_output_profile("match", 3) == 3
    assert resolve_output_profile("proxy", 3) == 0
    with pytest.raises(MediaError):
        resolve_output_profile("cinema", 3)


def test_square_clip_opens_centered_and_previews_both_views(tmp_path):
    source = tmp_path / "square.mov"
    _encode_prores(source, [np.full((64, 64, 3), 12000, dtype=np.uint16)])
    studio = Studio()
    opened = studio.open_clip(source)
    assert opened["square"] is True
    assert opened["center_x"] == pytest.approx(32.0)
    assert opened["center_y"] == pytest.approx(32.0)
    assert opened["radius"] == pytest.approx(32.0)
    params = {
        "path": str(source),
        "frame": "0",
        "center_x": "32",
        "center_y": "32",
        "radius": "32",
        "yaw": "0",
        "size": "8k",
        "rings": "1",
    }
    source_jpeg = studio.source_jpeg(params)
    unwrap_jpeg = studio.unwrap_jpeg(params)
    assert source_jpeg[:2] == b"\xff\xd8"
    assert unwrap_jpeg[:2] == b"\xff\xd8"
    assert len(unwrap_jpeg) > 500


def test_page_keeps_export_off_until_a_preview_exists(tmp_path):
    studio = Studio()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(studio))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
        assert 'id="export"' in page
        assert "disabled" in page
        assert "ProRes codec" in page
        health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health"))
        assert "ffmpeg" in health
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/export",
                    data=json.dumps({"path": str(tmp_path / "missing.mov"), "output": ""}).encode(),
                    headers={"Content-Type": "application/json"},
                )
            )
        assert caught.value.code == 400
    finally:
        httpd.shutdown()


def test_export_job_reports_progress(tmp_path, monkeypatch):
    source = tmp_path / "in.mov"
    output = tmp_path / "out.mov"
    _encode_prores(source, [np.full((32, 48, 3), 1000, dtype=np.uint16)])
    studio = Studio()

    def fake_render(args, movie, destination):
        args.on_progress(1, 1, "frame 1/1")
        destination.write_bytes(b"done")

    monkeypatch.setattr("studio.render_movie", fake_render)
    started = studio.start_export(
        {
            "path": str(source),
            "output": str(output),
            "center_x": 24,
            "center_y": 16,
            "radius": 15,
            "yaw": 0,
            "size": "8k",
            "profile": "hq",
        }
    )
    assert started["state"] == "running"
    for _ in range(50):
        job = studio.job_status()
        if job["state"] != "running":
            break
        threading.Event().wait(0.02)
    assert studio.job_status()["state"] == "done"
    assert output.read_bytes() == b"done"


def test_cli_profile_override_writes_proxy(tmp_path):
    source = tmp_path / "hq.mov"
    output = tmp_path / "proxy.mov"
    _encode_prores(source, [np.full((48, 64, 3), 2000, dtype=np.uint16)], profile=3)
    code = main(
        [
            str(source),
            "-o",
            str(output),
            "--profile",
            "proxy",
            "--size",
            "8k",
            "--preview-scale",
            "32",
            "--center-x",
            "32",
            "--center-y",
            "24",
            "--radius",
            "20",
        ]
    )
    assert code == 0
    info = probe_media(output)
    assert info.profile == 0
    assert info.codec_tag == "apco"
