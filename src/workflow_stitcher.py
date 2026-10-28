"""
Subroutine Clustering and Discontinuous Sequence Stitching.
Clusters segmented timeline tasks based on centroid embedding cosine affinity,
reconciles interleaved interruptions (A1 -> B -> A2), and stitches composite workflows.
"""

import json
import re

def format_compact_embeddings_json(manifest: SegmentedWorkflowManifest) -> str:
    """Serializes SegmentedWorkflowManifest to JSON with indent=2 while keeping centroid_embedding arrays on a single line."""
    raw_json = manifest.model_dump_json(indent=2)
    compact_json = re.sub(
        r'("centroid_embedding":\s*\[\s*)([\s\S]*?)(\s*\])',
        lambda m: f'"centroid_embedding": [{", ".join(x.strip() for x in m.group(2).split(",") if x.strip())}]',
        raw_json
    )
    return compact_json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.schemas import (
    TaskSegment,
    SubroutineCluster,
    StitchedSequence,
    SegmentedWorkflowManifest,
    TimelineManifest,
)
from src.boundary_detector import BoundaryDetector


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class WorkflowStitcher:
    """
    Performs subroutine clustering on task segments and stitches interrupted task sequences.
    """

    def __init__(self, cluster_sim_threshold: float = 0.85, max_interruption_gap_s: int = 120):
        """
        Args:
            cluster_sim_threshold: Minimum cosine similarity between centroid embeddings to group into the same cluster.
            max_interruption_gap_s: Maximum duration (in seconds) of an interleaved segment B for it to be considered an interruption.
        """
        self.cluster_sim_threshold = cluster_sim_threshold
        self.max_interruption_gap_s = max_interruption_gap_s

    def cluster_segments(self, segments: List[TaskSegment]) -> Tuple[List[TaskSegment], List[SubroutineCluster]]:
        """
        Clusters task segments by centroid embedding similarity and dominant application.
        Assigns cluster_id to each TaskSegment and generates SubroutineCluster records.
        """
        if not segments:
            return [], []

        clusters: List[SubroutineCluster] = []
        cluster_centroids: List[List[float]] = []

        for seg in segments:
            assigned_cluster_id = None
            best_sim = -1.0
            best_c_idx = -1

            if seg.centroid_embedding is not None:
                for c_idx, c_centroid in enumerate(cluster_centroids):
                    sim = cosine_similarity(seg.centroid_embedding, c_centroid)
                    # Check application compatibility
                    existing_cluster = clusters[c_idx]
                    app_match = (seg.primary_application in existing_cluster.primary_applications)

                    if sim > self.cluster_sim_threshold and (app_match or sim > 0.92):
                        if sim > best_sim:
                            best_sim = sim
                            best_c_idx = c_idx

            if best_c_idx >= 0:
                # Add to existing cluster
                assigned_cluster_id = clusters[best_c_idx].cluster_id
                clusters[best_c_idx].segment_ids.append(seg.segment_id)
                clusters[best_c_idx].total_duration_seconds += seg.duration_seconds
                if seg.primary_application not in clusters[best_c_idx].primary_applications:
                    clusters[best_c_idx].primary_applications.append(seg.primary_application)
            else:
                # Create a new cluster
                c_num = len(clusters) + 1
                assigned_cluster_id = f"cluster_{c_num:02d}"
                label = f"{seg.primary_application} Task Routine"
                new_cluster = SubroutineCluster(
                    cluster_id=assigned_cluster_id,
                    label=label,
                    segment_ids=[seg.segment_id],
                    total_duration_seconds=seg.duration_seconds,
                    primary_applications=[seg.primary_application] if seg.primary_application else [],
                    representative_summary=seg.summary_interaction,
                )
                clusters.append(new_cluster)
                cluster_centroids.append(seg.centroid_embedding if seg.centroid_embedding else [0.0] * 1536)

            seg.cluster_id = assigned_cluster_id

        return segments, clusters

    def stitch_discontinuous_sequences(
        self, segments: List[TaskSegment], clusters: List[SubroutineCluster]
    ) -> Tuple[List[TaskSegment], List[StitchedSequence]]:
        """
        Identifies interleaved interruptions (A1 -> B -> A2) and stitches them into unified macro-sequences.
        """
        if len(segments) < 2:
            single_seq = []
            if segments:
                single_seq.append(
                    StitchedSequence(
                        sequence_id="seq_01",
                        macro_task_label=segments[0].primary_application or "Continuous Routine",
                        primary_segment_ids=[segments[0].segment_id],
                        total_active_seconds=segments[0].duration_seconds,
                        total_interruption_seconds=0,
                        reconciliation_notes="Single contiguous task segment without interruptions.",
                    )
                )
            return segments, single_seq

        stitched_sequences: List[StitchedSequence] = []
        cluster_map = {c.cluster_id: c for c in clusters}

        # Look for pattern where segment i and i+2 share the same cluster, while i+1 is distinct
        i = 0
        visited_segments = set()

        while i < len(segments):
            seg_curr = segments[i]
            if seg_curr.segment_id in visited_segments:
                i += 1
                continue

            primary_ids = [seg_curr.segment_id]
            interruption_ids = []
            visited_segments.add(seg_curr.segment_id)

            curr_cluster = seg_curr.cluster_id
            active_duration = seg_curr.duration_seconds
            interruption_duration = 0
            reconciled_notes = []

            # Look forward for resumption of curr_cluster
            j = i + 1
            while j < len(segments):
                seg_next = segments[j]

                if seg_next.cluster_id == curr_cluster:
                    # Resumed task!
                    primary_ids.append(seg_next.segment_id)
                    active_duration += seg_next.duration_seconds
                    visited_segments.add(seg_next.segment_id)
                    reconciled_notes.append(
                        f"Resumed {seg_curr.primary_application} routine at segment {seg_next.segment_id}."
                    )
                    j += 1
                elif any(
                    segments[k].cluster_id == curr_cluster
                    for k in range(j + 1, min(j + 4, len(segments)))
                ):
                    # Find nearest resumption within lookahead window
                    next_k = next(
                        k for k in range(j + 1, min(j + 4, len(segments)))
                        if segments[k].cluster_id == curr_cluster
                    )
                    intervening_duration = sum(segments[m].duration_seconds for m in range(j, next_k))
                    if intervening_duration <= self.max_interruption_gap_s:
                        for m in range(j, next_k):
                            seg_inter = segments[m]
                            seg_inter.is_interruption = True
                            seg_inter.interrupted_segment_id = primary_ids[-1]
                            interruption_ids.append(seg_inter.segment_id)
                            interruption_duration += seg_inter.duration_seconds
                            visited_segments.add(seg_inter.segment_id)
                            reconciled_notes.append(
                                f"Interleaved {seg_inter.primary_application} ({seg_inter.duration_seconds}s) tagged as interruption."
                            )
                        j = next_k
                    else:
                        break
                else:
                    # Distinct next task began
                    break

            seq_id = f"seq_{len(stitched_sequences)+1:02d}"
            c_label = cluster_map.get(curr_cluster, SubroutineCluster(cluster_id="", label="Task")).label
            notes = "; ".join(reconciled_notes) if reconciled_notes else "Direct sequence execution."

            seq = StitchedSequence(
                sequence_id=seq_id,
                macro_task_label=c_label,
                primary_segment_ids=primary_ids,
                interleaved_segment_ids=interruption_ids,
                total_active_seconds=active_duration,
                total_interruption_seconds=interruption_duration,
                reconciliation_notes=notes,
            )
            stitched_sequences.append(seq)
            i = j

        return segments, stitched_sequences


def process_timeline_workflow(
    manifest_path: Path,
    output_path: Optional[Path] = None,
    diff_threshold: float = 0.5,
    sub_diff_threshold: float = 0.25,
    use_macro_only: bool = False,
) -> SegmentedWorkflowManifest:
    """
    Executes the complete Phase 3 pipeline:
    1. Detects breakpoints (using difference > 0.5 heuristic).
    2. Segments timeline slots into TaskSegments with centroid embeddings.
    3. Clusters recurring subroutines by cosine affinity.
    4. Reconciles interruptions and stitches macro-sequences.
    5. Saves and returns the SegmentedWorkflowManifest.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = TimelineManifest.model_validate(data)

    # 1. Boundary Detection & Segmentation
    detector = BoundaryDetector(diff_threshold=diff_threshold, sub_diff_threshold=sub_diff_threshold)
    raw_segments = detector.segment_timeline(manifest.slots, use_macro_only=use_macro_only)

    # 2. Subroutine Clustering & Discontinuous Sequence Stitching
    stitcher = WorkflowStitcher(cluster_sim_threshold=0.82)
    clustered_segments, clusters = stitcher.cluster_segments(raw_segments)
    final_segments, stitched_seqs = stitcher.stitch_discontinuous_sequences(clustered_segments, clusters)

    # 3. Assemble Manifest
    workflow_manifest = SegmentedWorkflowManifest(
        session_id=manifest.metadata.session_id,
        source_video=manifest.metadata.source_video,
        created_at=datetime.now(timezone.utc).isoformat(),
        difference_threshold=diff_threshold,
        total_segments=len(final_segments),
        total_clusters=len(clusters),
        total_stitched_sequences=len(stitched_seqs),
        segments=final_segments,
        clusters=clusters,
        stitched_sequences=stitched_seqs,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use compact format for serialization (centroid_embedding on single line)
        compact_content = format_compact_embeddings_json(workflow_manifest)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(compact_content)
        print(f"[OK] Saved SegmentedWorkflowManifest to {output_path}")

    return workflow_manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process timeline boundaries, subroutine clustering, and sequence stitching.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/solix_demo_test/timeline_manifest.json",
        help="Path to input timeline_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/solix_demo_test/segmented_workflow.json",
        help="Path to output segmented_workflow.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Breakpoint heuristic threshold (default: 0.5)",
    )
    parser.add_argument(
        "--macro-only",
        action="store_true",
        help="Only split at macro breakpoints (diff > threshold)",
    )
    args = parser.parse_args()

    process_timeline_workflow(
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        diff_threshold=args.threshold,
        use_macro_only=args.macro_only,
    )
