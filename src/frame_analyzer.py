"""
Perceptual frame analyzer for static frame detection and temporal visual diff scoring.
Uses lightweight downscaled luminance difference to identify idle/unchanged periods.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class FrameAnalyzer:
    def __init__(self, diff_threshold: float = 0.008, thumb_size: Tuple[int, int] = (64, 36)):
        """
        :param diff_threshold: Normalized mean absolute pixel delta below which a frame is considered static.
        :param thumb_size: Downsampled resolution for ultra-fast perceptual diffing.
        """
        self.diff_threshold = diff_threshold
        self.thumb_size = thumb_size

    def _get_fingerprint(self, image_path: Path) -> np.ndarray:
        """Loads and downsamples image to small normalized grayscale array."""
        with Image.open(image_path) as img:
            gray = img.convert("L").resize(self.thumb_size, Image.Resampling.BILINEAR)
            return np.asarray(gray, dtype=np.float32) / 255.0

    def analyze_sequence(self, frame_paths: List[Path]) -> List[Tuple[float, bool]]:
        """
        Analyzes a sequence of frames in order.
        Returns a list of (diff_score, is_static) tuples matching the input length.
        The first frame always has diff_score = 1.0 and is_static = False.
        """
        if not frame_paths:
            return []

        results: List[Tuple[float, bool]] = []
        prev_fingerprint: np.ndarray = None

        for idx, path in enumerate(frame_paths):
            try:
                curr_fingerprint = self._get_fingerprint(path)
            except Exception as e:
                logger.warning(f"Failed to read frame {path}: {e}")
                results.append((1.0, False))
                prev_fingerprint = None
                continue

            if prev_fingerprint is None:
                # First frame in session or chunk
                results.append((1.0, False))
            else:
                diff = float(np.mean(np.abs(curr_fingerprint - prev_fingerprint)))
                is_static = diff < self.diff_threshold
                results.append((round(diff, 5), is_static))

            prev_fingerprint = curr_fingerprint

        return results

