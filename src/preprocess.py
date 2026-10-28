"""
Main entry point for Phase 2: Ingestion & Frame-Audio Preprocessing.
Extracts 1 FPS frames, transcribes audio, identifies static intervals,
and produces a synchronized timeline manifest for arbitrary duration sessions.
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Optional

from src.config import (
    BASE_DIR,
    DEFAULT_FPS,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_CHUNK_DURATION_SEC,
    STATIC_DIFF_THRESHOLD,
    FRAMES_DIR,
    AUDIO_DIR,
    PROCESSED_DIR
)
from src.media_extractor import MediaExtractor
from src.audio_transcriber import AudioTranscriber
from src.frame_analyzer import FrameAnalyzer
from src.timeline_aligner import TimelineAligner
from src.tier1_descriptor import enrich_manifest
from src.schemas import SessionMetadata

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("preprocess")

def run_preprocessing(
    video_path: Path,
    session_id: Optional[str] = None,
    fps: int = DEFAULT_FPS,
    diff_threshold: float = STATIC_DIFF_THRESHOLD,
    skip_transcription: bool = False,
    mock_transcription: bool = False,
    enrich_descriptors: bool = True,
    mock_vlm: bool = False,
    force_all_frames: bool = False
) -> Path:
    """
    Executes Phase 2 preprocessing on a video recording and optionally
    enriches with gpt-4o-mini descriptors, raw text, and vector embeddings.
    Returns path to the generated timeline_manifest.json.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Generate session ID if not provided
    if not session_id:
        session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    session_frames_dir = FRAMES_DIR / session_id
    session_audio_dir = AUDIO_DIR / session_id
    session_processed_dir = PROCESSED_DIR / session_id

    session_frames_dir.mkdir(parents=True, exist_ok=True)
    session_audio_dir.mkdir(parents=True, exist_ok=True)
    session_processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Starting Phase 2 Ingestion for Session: {session_id}")
    logger.info(f"Source video: {video_path}")
    logger.info("=" * 60)

    # 1. Probe video
    extractor = MediaExtractor()
    probe_info = extractor.probe(video_path)
    duration_sec = probe_info["duration"]
    has_audio = probe_info["has_audio"]

    logger.info(
        f"Probed media: duration={duration_sec:.1f}s, resolution={probe_info['width']}x{probe_info['height']}, "
        f"audio_track={has_audio}"
    )

    # 2. Extract frames at target FPS
    logger.info(f"Extracting frames at {fps} FPS to {session_frames_dir}...")
    frame_paths = extractor.extract_frames(
        video_path=video_path,
        output_dir=session_frames_dir,
        fps=fps,
        scale="1920:1080"
    )
    total_frames = len(frame_paths)
    logger.info(f"Successfully extracted {total_frames} frames.")

    # 3. Audio Extraction & Transcription
    audio_segments = []
    if has_audio and not skip_transcription:
        audio_wav = session_audio_dir / "audio_track.wav"
        logger.info(f"Extracting 16kHz mono audio to {audio_wav}...")
        extracted_wav = extractor.extract_audio(
            video_path=video_path,
            output_wav=audio_wav,
            sample_rate=DEFAULT_AUDIO_SAMPLE_RATE
        )

        if extracted_wav:
            if mock_transcription:
                logger.info("Mock transcription requested: using placeholder transcript.")
                from src.schemas import AudioSegment
                audio_segments = [
                    AudioSegment(start=0.0, end=min(3.0, duration_sec), text="Mock audio narration test")
                ]
            else:
                try:
                    transcriber = AudioTranscriber()
                    audio_segments = transcriber.transcribe(extracted_wav)
                except Exception as e:
                    logger.warning(f"Audio transcription encountered error (continuing without audio): {e}")
                    audio_segments = []
    elif not has_audio:
        logger.info("Source video has no audio track; proceeding with visual frames only.")
    else:
        logger.info("Audio transcription skipped per user flag.")

    # 4. Frame analysis (perceptual diff & static frame detection)
    logger.info("Analyzing frame sequence for static/idle detection...")
    analyzer = FrameAnalyzer(diff_threshold=diff_threshold)
    frame_metrics = analyzer.analyze_sequence(frame_paths)
    static_count = sum(1 for _, is_static in frame_metrics if is_static)
    logger.info(f"Detected {static_count} static frames ({static_count / max(1, total_frames) * 100:.1f}%).")

    # 5. Temporal Alignment
    logger.info("Aligning frames and audio into timeline manifest...")
    metadata = SessionMetadata(
        session_id=session_id,
        source_video=str(video_path),
        duration_seconds=duration_sec,
        total_frames=total_frames,
        fps=fps,
        has_audio_track=has_audio,
        created_at=datetime.now(timezone.utc).isoformat()
    )

    manifest = TimelineAligner.align(
        frame_paths=frame_paths,
        frame_metrics=frame_metrics,
        audio_segments=audio_segments,
        metadata=metadata,
        base_dir=BASE_DIR
    )

    # 6. Tier 1: Multimodal Descriptor Extraction & Embedding Generation
    if enrich_descriptors:
        logger.info("Enriching manifest with gpt-4o-mini descriptors, raw text, and vector embeddings...")
        manifest = enrich_manifest(
            manifest=manifest,
            base_dir=BASE_DIR,
            use_cache_for_static=not force_all_frames,
            mock=mock_vlm
        )
    else:
        logger.info("Descriptor enrichment skipped per user flag.")

    # 7. Save manifest
    manifest_path = session_processed_dir / "timeline_manifest.json"
    TimelineAligner.save_manifest(manifest, manifest_path)

    logger.info("=" * 60)
    logger.info("[SUCCESS] Preprocessing & Enrichment completed successfully!")
    logger.info(f"Manifest output: {manifest_path}")
    logger.info(f"Total slots:     {len(manifest.slots)}")
    logger.info(f"Static slots:    {static_count}")
    logger.info(f"Speech segments: {len(audio_segments)}")
    if enrich_descriptors:
        enriched_count = sum(1 for s in manifest.slots if s.descriptor is not None)
        embedded_count = sum(1 for s in manifest.slots if s.embedding is not None)
        logger.info(f"Descriptors:     {enriched_count}/{len(manifest.slots)}")
        logger.info(f"Embeddings:      {embedded_count}/{len(manifest.slots)}")
    logger.info("=" * 60)

    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="Ingest & Preprocess Video Capture for Workflow Analysis")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input MP4/MKV video recording")
    parser.add_argument("--session-id", "-s", type=str, default=None, help="Custom session ID string")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frame extraction rate (default: 1 FPS)")
    parser.add_argument("--diff-threshold", type=float, default=STATIC_DIFF_THRESHOLD, help="Static frame diff threshold")
    parser.add_argument("--skip-audio", action="store_true", help="Skip Whisper audio transcription")
    parser.add_argument("--mock-audio", action="store_true", help="Generate mock transcript without API call")
    parser.add_argument("--no-enrich", action="store_true", help="Skip gpt-4o-mini descriptor and embedding enrichment")
    parser.add_argument("--mock-vlm", action="store_true", help="Generate mock descriptors/embeddings without API calls")
    parser.add_argument("--force-all-frames", action="store_true", help="Bypass static frame caching for VLM inference")

    args = parser.parse_args()
    video_path = Path(args.video).resolve()

    try:
        manifest_path = run_preprocessing(
            video_path=video_path,
            session_id=args.session_id,
            fps=args.fps,
            diff_threshold=args.diff_threshold,
            skip_transcription=args.skip_audio,
            mock_transcription=args.mock_audio,
            enrich_descriptors=not args.no_enrich,
            mock_vlm=args.mock_vlm,
            force_all_frames=args.force_all_frames
        )
        print(f"\nManifest ready at: {manifest_path}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

