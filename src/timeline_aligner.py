"""
Timeline alignment engine: aligns 1 FPS visual frames, static detection metrics,
and timestamped speech segments into a unified master timeline manifest.
"""

import json
import logging
from pathlib import Path
import re
from typing import List, Tuple
from src.schemas import AudioSegment, SessionMetadata, TimelineManifest, TimelineSlot

logger = logging.getLogger(__name__)

def format_compact_embeddings_json(manifest: TimelineManifest) -> str:
    """Serializes manifest to JSON with indent=2 while keeping embedding arrays on a single line."""
    raw_json = manifest.model_dump_json(indent=2)
    # Collapse multi-line arrays for "embedding": [ ... ]
    compact_json = re.sub(
        r'("embedding":\s*\[\s*)([\s\S]*?)(\s*\])',
        lambda m: f'"embedding": [{", ".join(x.strip() for x in m.group(2).split(",") if x.strip())}]',
        raw_json
    )
    return compact_json

class TimelineAligner:
    @staticmethod
    def align(
        frame_paths: List[Path],
        frame_metrics: List[Tuple[float, bool]],
        audio_segments: List[AudioSegment],
        metadata: SessionMetadata,
        base_dir: Path
    ) -> TimelineManifest:
        """
        Aligns extracted frames and audio intervals into second-by-second TimelineSlots.
        """
        slots: List[TimelineSlot] = []

        total_frames = len(frame_paths)
        if len(frame_metrics) != total_frames:
            raise ValueError(
                f"Mismatch between frames count ({total_frames}) and metrics count ({len(frame_metrics)})"
            )

        for second_idx, frame_path in enumerate(frame_paths):
            diff_score, is_static = frame_metrics[second_idx]

            # Interval for this slot: [second_idx, second_idx + 1.0)
            slot_start = float(second_idx)
            slot_end = float(second_idx + 1.0)

            # Match any audio segment that overlaps with this second window
            overlapping_texts: List[str] = []
            for seg in audio_segments:
                if seg.start < slot_end and seg.end > slot_start:
                    if seg.text and seg.text not in overlapping_texts:
                        overlapping_texts.append(seg.text)

            has_audio = len(overlapping_texts) > 0
            audio_text = " ".join(overlapping_texts)

            # Store relative path for portability
            try:
                rel_path = str(frame_path.relative_to(base_dir))
            except ValueError:
                rel_path = str(frame_path)

            slots.append(TimelineSlot(
                timestamp=second_idx,
                frame_path=rel_path,
                has_audio=has_audio,
                audio_text=audio_text,
                is_static=is_static,
                diff_score=diff_score
            ))

        manifest = TimelineManifest(
            metadata=metadata,
            audio_segments=audio_segments,
            slots=slots
        )

        return manifest

    @staticmethod
    def save_manifest(manifest: TimelineManifest, output_json: Path) -> Path:
        """Serializes TimelineManifest to a pretty-printed JSON file with compact single-line embeddings."""
        output_json.parent.mkdir(parents=True, exist_ok=True)
        formatted_json = format_compact_embeddings_json(manifest)
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(formatted_json)
        logger.info(f"Saved timeline manifest to {output_json} ({len(manifest.slots)} slots).")
        return output_json

