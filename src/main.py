"""
Master End-to-End Orchestrator for the Agentic Workflow Capture & Analysis System.

Executes the complete multimodal pipeline:
  1. Ingestion & Preprocessing (FFmpeg 1 FPS frames, audio track, static frame analysis)
  2. Tier 1 Multimodal Extraction & Embeddings (gpt-4o-mini Vision + text-embedding-3-small)
  3. Semantic Derivative Analysis & Plotting (Consecutive frame dot products & diffs)
  4. Task Boundary Detection, Subroutine Clustering & Sequence Stitching (Breakpoint heuristic > 0.5)
  5. Tier 2 Workflow Synthesis (gpt-4o-mini business logic, Mermaid flowchart, friction ledger, automation blueprints)
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from datetime import datetime

# Local imports
from src.preprocess import run_preprocessing
from src.plot_derivatives import generate_plot
from src.workflow_stitcher import process_timeline_workflow
from src.tier2_synthesis import run_synthesis_pipeline
from src.automation_generator import generate_automation_package
from src.dashboard_generator import generate_dashboard
from src.config import BASE_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_full_pipeline(
    video_path: Path,
    session_id: str = None,
    diff_threshold: float = 0.5,
    skip_audio: bool = False,
    mock: bool = False,
    force_all_frames: bool = False,
    artifact_dir: Path = None,
) -> dict:
    """
    Executes the full end-to-end pipeline on an input video recording.
    """
    start_time = datetime.now()

    if not video_path.exists():
        logger.error(f"Input video not found: {video_path}")
        sys.exit(1)

    if not session_id:
        session_id = video_path.stem

    output_dir = PROCESSED_DIR / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"STARTING FULL END-TO-END PIPELINE: {session_id}")
    logger.info(f"Source video:     {video_path}")
    logger.info(f"Target directory: {output_dir}")
    logger.info(f"Breakpoint heuristic threshold: {diff_threshold}")
    logger.info(f"Mock mode:        {mock}")
    logger.info("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1 & 2: Ingestion, Tier 1 Descriptors & Embeddings
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 1 & 2: Ingestion & Tier 1 Multimodal Descriptors & Embeddings...")
    manifest_path = run_preprocessing(
        video_path=video_path,
        session_id=session_id,
        skip_transcription=skip_audio,
        mock_transcription=mock,
        enrich_descriptors=True,
        mock_vlm=mock,
        force_all_frames=force_all_frames,
    )
    logger.info(f"[OK] Timeline manifest generated: {manifest_path}")

    # -------------------------------------------------------------
    # STAGE 3: Semantic Derivative Analysis & Breakpoint Plotting
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 3: Semantic Derivative Analysis & Breakpoint Plotting...")
    plot_png_path = output_dir / "consecutive_dot_products.png"
    art_plot_path = (artifact_dir / "consecutive_dot_products.png") if artifact_dir else None

    generate_plot(
        manifest_path=manifest_path,
        output_png=plot_png_path,
        artifact_png=art_plot_path,
        diff_threshold=diff_threshold,
    )
    logger.info(f"[OK] Derivative trajectory plot generated: {plot_png_path}")

    # -------------------------------------------------------------
    # STAGE 4: Task Boundaries, Clustering & Sequence Stitching
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 4: Boundaries, Subroutine Clustering & Discontinuous Stitching...")
    segmented_json_path = output_dir / "segmented_workflow.json"

    workflow_manifest = process_timeline_workflow(
        manifest_path=manifest_path,
        output_path=segmented_json_path,
        diff_threshold=diff_threshold,
        use_macro_only=False,
    )
    logger.info(
        f"[OK] Clustered {workflow_manifest.total_segments} segments into "
        f"{workflow_manifest.total_clusters} clusters & {workflow_manifest.total_stitched_sequences} stitched sequences: {segmented_json_path}"
    )

    # -------------------------------------------------------------
    # STAGE 5: Tier 2 Business Logic & Automation Synthesis
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 5: Tier 2 Workflow Synthesis & Automation Blueprinting (gpt-4o-mini)...")
    synthesis_json_path = output_dir / "workflow_synthesis.json"
    synthesis_report_path = output_dir / "workflow_synthesis_report.md"
    art_report_path = (artifact_dir / "workflow_synthesis_report.md") if artifact_dir else None

    synthesis_report = run_synthesis_pipeline(
        input_manifest_path=segmented_json_path,
        output_json_path=synthesis_json_path,
        report_markdown_path=synthesis_report_path,
        mock=mock,
    )

    if art_report_path:
        from src.tier2_synthesis import format_markdown_report
        art_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(art_report_path, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(synthesis_report))
        logger.info(f"[OK] Saved copy to artifact directory: {art_report_path}")

    # -------------------------------------------------------------
    # STAGE 6: Standalone Python Automation Generation
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 6: Generating Standalone Python Automations...")
    generated_automations = generate_automation_package(synthesis_json_path)
    logger.info(f"[OK] Generated {len(generated_automations)} automation scripts in src/automations/")

    # -------------------------------------------------------------
    # STAGE 7: Interactive Verification Dashboard Generation
    # -------------------------------------------------------------
    logger.info("\n>>> STAGE 7: Generating Interactive Visual Verification Dashboard...")
    dashboard_path = generate_dashboard(session_id=session_id)
    logger.info(f"[OK] Generated Interactive Dashboard: {dashboard_path}")

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("=" * 70)
    logger.info(f"[SUCCESS] Full Pipeline Completed in {elapsed:.1f}s")
    logger.info("Deliverables Generated:")
    logger.info(f"  1. Timeline Manifest:   {manifest_path}")
    logger.info(f"  2. Breakpoint Plot:     {plot_png_path}")
    logger.info(f"  3. Segmented Workflow:  {segmented_json_path}")
    logger.info(f"  4. Synthesis JSON:      {synthesis_json_path}")
    logger.info(f"  5. Synthesis Report:    {synthesis_report_path}")
    logger.info(f"  6. Python Automations:  {[p.name for p in generated_automations]}")
    logger.info(f"  7. Visual Dashboard:    {dashboard_path}")
    logger.info("=" * 70)

    return {
        "manifest_path": manifest_path,
        "plot_path": plot_png_path,
        "segmented_path": segmented_json_path,
        "synthesis_json_path": synthesis_json_path,
        "synthesis_report_path": synthesis_report_path,
        "automations": generated_automations,
        "dashboard_path": dashboard_path,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end multimodal workflow capture, clustering, and synthesis pipeline."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="tests/fixtures/solix_demo_test.mp4",
        help="Path to source screen/audio recording video file (default: tests/fixtures/solix_demo_test.mp4)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Custom session identifier (default: inferred from video filename)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Task breakpoint difference heuristic threshold (default: 0.5)",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip Whisper audio transcription step",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in fast offline deterministic mock mode without API costs",
    )
    parser.add_argument(
        "--force-all-frames",
        action="store_true",
        help="Force VLM evaluation of all frames (disables static frame caching)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="Optional path to artifact directory to copy report and plot",
    )

    args = parser.parse_args()

    art_dir = Path(args.artifact_dir) if args.artifact_dir else None

    run_full_pipeline(
        video_path=Path(args.video),
        session_id=args.session_id,
        diff_threshold=args.threshold,
        skip_audio=args.skip_audio,
        mock=args.mock,
        force_all_frames=args.force_all_frames,
        artifact_dir=art_dir,
    )


if __name__ == "__main__":
    main()
