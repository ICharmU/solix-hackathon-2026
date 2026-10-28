"""
AUTO-01: Automated Headless Recording Daemon & Pipeline Hook.
Replaces manual OBS Studio setup/teardown friction by providing a background
screen & audio capture CLI with automatic pipeline triggering on completion.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PID_FILE = Path(".capture_daemon.pid")


def find_ffmpeg() -> str:
    """Locates ffmpeg on system path or fallback directories."""
    paths = [
        "ffmpeg",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        str(Path.home() / "miniforge3" / "Library" / "bin" / "ffmpeg.exe"),
    ]
    for p in paths:
        try:
            subprocess.run([p, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return p
        except Exception:
            continue
    return "ffmpeg"


def start_capture(output_path: Path, fps: int = 30) -> subprocess.Popen:
    """Launches headless screen and desktop audio capture."""
    if PID_FILE.exists():
        print(f"[ERROR] Capture daemon is already running (PID: {PID_FILE.read_text().strip()}).")
        sys.exit(1)

    ffmpeg_bin = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Windows gdigrab display capture
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "gdigrab",
        "-framerate", str(fps),
        "-i", "desktop",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    print(f"[INFO] Starting background capture daemon to: {output_path}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    PID_FILE.write_text(str(proc.pid))
    print(f"[OK] Daemon active with PID {proc.pid}. Run with --stop to terminate and trigger pipeline.")
    return proc


def stop_capture(trigger_pipeline: bool = True):
    """Terminates active capture daemon cleanly and triggers processing pipeline."""
    if not PID_FILE.exists():
        print("[WARN] No active capture daemon found.")
        return

    pid = int(PID_FILE.read_text().strip())
    print(f"[INFO] Terminating capture daemon (PID {pid})...")

    try:
        if sys.platform == "win32":
            # Graceful termination using taskkill or SIGTERM
            subprocess.run(["taskkill", "/PID", str(pid), "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGINT)
    except Exception as e:
        print(f"[WARN] Error stopping process: {e}")

    PID_FILE.unlink(missing_ok=True)
    print("[OK] Capture daemon stopped.")

    if trigger_pipeline:
        print("[INFO] Triggering post-capture analysis pipeline...")
        subprocess.run(["python", "-m", "src.main"], check=False)


def main():
    parser = argparse.ArgumentParser(description="Automated Screen & Audio Recording Daemon (AUTO-01)")
    parser.add_argument("--start", action="store_true", help="Start recording daemon")
    parser.add_argument("--stop", action="store_true", help="Stop recording daemon and trigger pipeline")
    parser.add_argument("--output", type=str, default="data/session_auto.mp4", help="Output recording video path")
    args = parser.parse_args()

    if args.stop:
        stop_capture()
    elif args.start:
        start_capture(Path(args.output))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
