"""
Media extraction module using FFmpeg and FFprobe.
Extracts 1 FPS visual frames and 16kHz mono audio tracks with arbitrary duration support.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class MediaExtractor:
    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.ffmpeg = ffmpeg_bin
        self.ffprobe = ffprobe_bin

    def probe(self, video_path: Path) -> Dict[str, Any]:
        """Probes video file using ffprobe and returns stream & format information."""
        if not video_path.exists():
            raise FileNotFoundError(f"Input video file not found: {video_path}")

        cmd = [
            self.ffprobe,
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate",
            "-of", "json",
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFprobe execution failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse FFprobe JSON: {e}")

        format_info = probe_data.get("format", {})
        streams = probe_data.get("streams", [])

        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        duration = float(format_info.get("duration", 0.0))

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        width = int(video_stream.get("width", 0)) if video_stream else 0
        height = int(video_stream.get("height", 0)) if video_stream else 0

        return {
            "duration": duration,
            "has_video": has_video,
            "has_audio": has_audio,
            "width": width,
            "height": height,
            "streams": streams
        }

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        fps: int = 1,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        scale: str = "1920:1080"
    ) -> List[Path]:
        """
        Extracts frames at given FPS (default 1 FPS).
        Supports chunked extraction via start_sec and duration_sec for arbitrary recording lengths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        filename_pattern = output_dir / "frame_%06d.jpg"

        cmd = [self.ffmpeg, "-y"]

        if start_sec is not None:
            cmd.extend(["-ss", f"{start_sec:.3f}"])

        cmd.extend(["-i", str(video_path)])

        if duration_sec is not None:
            cmd.extend(["-t", f"{duration_sec:.3f}"])

        # vf: set fps, scale preserving aspect ratio with letterbox/padding if needed
        vf_filter = f"fps={fps}"
        if scale:
            vf_filter += f",scale={scale}:force_original_aspect_ratio=decrease,pad={scale}:(ow-iw)/2:(oh-ih)/2"

        cmd.extend([
            "-vf", vf_filter,
            "-qscale:v", "2",
            str(filename_pattern)
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg frame extraction failed: {e.stderr}")

        # Return sorted list of extracted frame paths
        frames = sorted(output_dir.glob("frame_*.jpg"))
        return frames

    def extract_audio(
        self,
        video_path: Path,
        output_wav: Path,
        sample_rate: int = 16000,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None
    ) -> Optional[Path]:
        """
        Extracts audio as a 16kHz mono PCM WAV file suitable for Whisper.
        Returns None if video has no audio stream.
        """
        info = self.probe(video_path)
        if not info["has_audio"]:
            logger.info("Video does not contain an audio track. Skipping audio extraction.")
            return None

        output_wav.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.ffmpeg, "-y"]
        if start_sec is not None:
            cmd.extend(["-ss", f"{start_sec:.3f}"])

        cmd.extend(["-i", str(video_path)])

        if duration_sec is not None:
            cmd.extend(["-t", f"{duration_sec:.3f}"])

        cmd.extend([
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(output_wav)
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")

        if output_wav.exists() and output_wav.stat().st_size > 0:
            return output_wav
        return None

