"""Small desktop wrapper around the skinthefish command line.

The window shells out to ``main.py`` so the render path is the same one the
batch command uses. It needs a display and tkinter, which is part of the
standard library on most workstation installs and is absent on some servers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(values: dict[str, str], python: str | None = None) -> list[str]:
    """Turn the form fields into a ``main.py`` command."""
    source = values.get("input", "").strip()
    if not source:
        raise ValueError("choose an input movie or folder")
    main_path = Path(__file__).resolve().parent / "main.py"
    command = [python or sys.executable, str(main_path), source]
    output = values.get("output", "").strip()
    if output:
        command += ["-o", output]
    output_dir = values.get("output_dir", "").strip()
    if output_dir:
        command += ["--output-dir", output_dir]
    command += ["--interp", values.get("interp") or "lanczos4"]
    for flag, key in (
        ("--center-x", "center_x"),
        ("--center-y", "center_y"),
        ("--radius", "radius"),
        ("--yaw-offset", "yaw_offset"),
        ("--theta-max", "theta_max"),
        ("--preview-scale", "preview_scale"),
        ("--calibration", "calibration"),
    ):
        text = values.get(key, "").strip()
        if text:
            command += [flag, text]
    if values.get("calibrate") == "1":
        command.append("--calibrate")
    if values.get("overwrite") == "1":
        command.append("--overwrite")
    return command


def launch(args: argparse.Namespace | None = None) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError as exc:
        raise SystemExit(
            "tkinter is not installed. Render from the command line with python main.py, "
            "or install the python3-tk package."
        ) from exc

    root = tk.Tk()
    root.title("skinthefish — HAL 250 unwrap")
    root.geometry("760x640")
    fields: dict[str, tk.StringVar] = {
        "input": tk.StringVar(value=str(args.input) if args and args.input else ""),
        "output": tk.StringVar(value=str(args.output) if args and args.output else ""),
        "output_dir": tk.StringVar(value=str(args.output_dir) if args and args.output_dir else ""),
        "interp": tk.StringVar(value=getattr(args, "interp", "lanczos4")),
        "center_x": tk.StringVar(value=_optional(args, "center_x")),
        "center_y": tk.StringVar(value=_optional(args, "center_y")),
        "radius": tk.StringVar(value=_optional(args, "radius")),
        "yaw_offset": tk.StringVar(value=_optional(args, "yaw_offset")),
        "theta_max": tk.StringVar(value=_optional(args, "theta_max")),
        "preview_scale": tk.StringVar(value=str(getattr(args, "preview_scale", 1) or 1)),
        "calibration": tk.StringVar(value=str(args.calibration) if args and args.calibration else ""),
    }
    calibrate = tk.StringVar(value="1" if args and args.calibrate else "0")
    overwrite = tk.StringVar(value="1" if args and args.overwrite else "0")

    form = ttk.Frame(root, padding=12)
    form.pack(fill="x")
    rows = [
        ("Input", "input"),
        ("Output movie", "output"),
        ("Output folder", "output_dir"),
        ("Calibration JSON", "calibration"),
        ("Center X", "center_x"),
        ("Center Y", "center_y"),
        ("Radius", "radius"),
        ("Yaw offset", "yaw_offset"),
        ("Theta max", "theta_max"),
        ("Preview scale", "preview_scale"),
    ]
    for row, (label, key) in enumerate(rows):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=fields[key], width=64).grid(row=row, column=1, sticky="ew", pady=2)
    form.columnconfigure(1, weight=1)

    picks = ttk.Frame(root, padding=(12, 0))
    picks.pack(fill="x")
    ttk.Label(picks, text="Remap kernel").grid(row=0, column=0, sticky="w")
    ttk.Combobox(picks, textvariable=fields["interp"], values=("lanczos4", "cubic"), width=10).grid(row=0, column=1, padx=4)
    ttk.Label(picks, text="Upscale is Lanczos-4 to 16:9. No sharpen, no pad.").grid(row=0, column=2, sticky="w", padx=8)

    checks = ttk.Frame(root, padding=12)
    checks.pack(fill="x")
    ttk.Checkbutton(checks, text="Calibrate still only", variable=calibrate, onvalue="1", offvalue="0").pack(side="left")
    ttk.Checkbutton(checks, text="Overwrite", variable=overwrite, onvalue="1", offvalue="0").pack(side="left", padx=12)

    buttons = ttk.Frame(root, padding=(12, 0))
    buttons.pack(fill="x")

    log = tk.Text(root, height=16, wrap="word")
    log.pack(fill="both", expand=True, padx=12, pady=12)

    def browse_input() -> None:
        selected = filedialog.askopenfilename(title="Input movie")
        if selected:
            fields["input"].set(selected)

    def run_job() -> None:
        payload = {key: var.get() for key, var in fields.items()}
        payload["calibrate"] = calibrate.get()
        payload["overwrite"] = overwrite.get()
        try:
            command = build_command(payload)
        except ValueError as exc:
            log.insert("end", f"{exc}\n")
            return
        log.insert("end", "$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        log.see("end")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def pump() -> None:
            assert process.stdout is not None
            chunk = process.stdout.readline()
            if chunk:
                log.insert("end", chunk)
                log.see("end")
                root.after(10, pump)
                return
            code = process.wait()
            log.insert("end", f"\nexit {code}\n")
            log.see("end")

        root.after(10, pump)

    ttk.Button(buttons, text="Browse input", command=browse_input).pack(side="left")
    ttk.Button(buttons, text="Run", command=run_job).pack(side="left", padx=8)
    root.mainloop()


def _optional(args: argparse.Namespace | None, name: str) -> str:
    if args is None:
        return ""
    value = getattr(args, name, None)
    return "" if value is None else str(value)


if __name__ == "__main__":
    prefill = None
    if len(sys.argv) > 1:
        from main import build_parser

        prefill = build_parser().parse_args()
    try:
        launch(prefill)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc.code, file=sys.stderr)
            raise
