"""Pick a PyTorch CUDA wheel this NVIDIA driver can load.

A newer driver runs an older wheel. An older driver cannot run a newer wheel.
CUDA 12.8 wheels cover driver 12.8 and above, including the A4500. Drivers
older than that get the CUDA 12.4 wheel.
"""

from __future__ import annotations

import re
import subprocess


def index_for_cuda(major: int, minor: int) -> str:
    if (major, minor) >= (12, 8):
        return "cu128"
    return "cu124"


def index_from_nvidia_smi() -> str:
    try:
        text = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return "cu128"
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    if not match:
        return "cu128"
    return index_for_cuda(int(match.group(1)), int(match.group(2)))


if __name__ == "__main__":
    print(index_from_nvidia_smi())
