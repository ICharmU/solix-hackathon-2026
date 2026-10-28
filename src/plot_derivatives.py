"""
Plot consecutive frame embedding difference and dot products over time to identify workflow state transitions and task breakpoints.
"""

import os
import sys
from pathlib import Path

# Configure matplotlib cache directory locally within workspace
mpl_dir = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import json
import argparse
import traceback
from typing import List, Dict, Any, Tuple
import numpy as np


def compute_consecutive_metrics(
    slots: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Computes dot products, embedding differences (1 - dot), visual diff scores, and metadata.
    """
    timestamps = []
    dot_products = []
    differences = []
    diff_scores = []
    apps = []
    actions = []

    for i in range(1, len(slots)):
        prev_slot = slots[i - 1]
        curr_slot = slots[i]

        t = curr_slot.get("timestamp", i)
        timestamps.append(t)
        diff_scores.append(curr_slot.get("diff_score", 0.0))

        app = curr_slot.get("descriptor", {}).get("active_application", "Unknown")
        action = curr_slot.get("descriptor", {}).get("observed_interaction", "")
        apps.append(app)
        actions.append(action)

        vec_prev = np.array(prev_slot.get("embedding", []), dtype=np.float32)
        vec_curr = np.array(curr_slot.get("embedding", []), dtype=np.float32)

        if len(vec_prev) == 0 or len(vec_curr) == 0:
            dot_products.append(1.0)
            differences.append(0.0)
            continue

        norm_prev = np.linalg.norm(vec_prev)
        norm_curr = np.linalg.norm(vec_curr)

        if norm_prev > 1e-6 and norm_curr > 1e-6:
            dot = float(np.dot(vec_prev, vec_curr) / (norm_prev * norm_curr))
        else:
            dot = float(np.dot(vec_prev, vec_curr))

        dot = max(-1.0, min(1.0, dot))
        diff = 1.0 - dot

        dot_products.append(dot)
        differences.append(diff)

    return (
        np.array(timestamps),
        np.array(dot_products),
        np.array(differences),
        np.array(diff_scores),
        apps,
        actions,
    )


def generate_plot(
    manifest_path: Path,
    output_png: Path,
    artifact_png: Path = None,
    diff_threshold: float = 0.5,
):
    """
    Generates a dual-panel visualization showing:
    1. Consecutive frame embedding difference Δ(t) = 1 - (e_t · e_{t-1}) and dot product.
       Denotes distinct task breakpoints where difference > diff_threshold (heuristic).
    2. Visual frame diff score over time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session_id = data.get("metadata", {}).get("session_id", "demo_session")
    slots = data.get("slots", [])

    if len(slots) < 2:
        print(f"Error: Need at least 2 slots to compute consecutive metrics, found {len(slots)}")
        return

    timestamps, dot_products, differences, diff_scores, apps, actions = compute_consecutive_metrics(slots)

    # Heuristic: Breakpoints where embedding difference > diff_threshold (e.g. > 0.5)
    breakpoint_indices = np.where(differences > diff_threshold)[0]
    breakpoint_times = timestamps[breakpoint_indices]

    # Set up styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # --- Top Subplot: Embedding Difference Δ(t) and Dot Product ---
    ax1.plot(
        timestamps,
        differences,
        color="#2563eb",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label=r"Embedding Difference $\Delta(t) = 1 - (e_t \cdot e_{t-1})$",
        zorder=3,
    )

    # Highlight heuristic breakpoints (difference > 0.5)
    ax1.scatter(
        timestamps[breakpoint_indices],
        differences[breakpoint_indices],
        color="#dc2626",
        marker="D",
        s=120,
        zorder=6,
        edgecolors="black",
        linewidths=1.5,
        label=f"Distinct Task Breakpoint (Difference > {diff_threshold:.2f})",
    )

    # Horizontal threshold line
    ax1.axhline(
        y=diff_threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        label=f"Task Breakpoint Threshold ($\\Delta > {diff_threshold:.2f}$)",
        zorder=4,
    )

    # Secondary y-axis for Dot Product
    ax1_sec = ax1.twinx()
    ax1_sec.set_ylabel(r"Dot Product ($e_t \cdot e_{t-1}$)", fontsize=11, fontweight="bold", color="#6b7280")
    ax1_sec.set_ylim(1.05, -0.05)  # Inverted so 1.0 aligns with difference 0.0
    ax1_sec.tick_params(axis="y", labelcolor="#6b7280")
    ax1_sec.grid(False)

    # Application background shading
    current_app = apps[0]
    app_start_t = timestamps[0]
    app_colors = {
        "OBS Studio": "#e0e7ff",
        "Visual Studio Code": "#dcfce7",
        "Google Chrome": "#fef3c7",
        "VS Code": "#dcfce7",
        "Chrome": "#fef3c7",
    }

    for i in range(1, len(timestamps)):
        if apps[i] != current_app or i == len(timestamps) - 1:
            end_t = timestamps[i] if apps[i] != current_app else timestamps[i] + 1
            color = app_colors.get(current_app, "#f3f4f6")
            ax1.axvspan(app_start_t - 0.5, end_t - 0.5, color=color, alpha=0.4, zorder=1)
            mid_t = (app_start_t + end_t) / 2
            ax1.text(
                mid_t,
                0.04,
                current_app,
                ha="center",
                va="bottom",
                fontsize=9.0,
                fontweight="bold",
                color="#374151",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d1d5db", alpha=0.85),
                zorder=4,
            )
            current_app = apps[i]
            app_start_t = timestamps[i]

    # Draw vertical breakpoint lines and task segment labels
    task_boundaries = [0] + list(breakpoint_times) + [timestamps[-1]]
    task_boundaries = sorted(list(set(task_boundaries)))

    for b_idx in breakpoint_indices:
        t = timestamps[b_idx]
        diff_val = differences[b_idx]
        dot_val = dot_products[b_idx]
        prev_app = apps[b_idx - 1] if b_idx > 0 else "Start"
        next_app = apps[b_idx]

        # Vertical line through both subplots
        ax1.axvline(x=t, color="#dc2626", linestyle=":", linewidth=2.0, alpha=0.85, zorder=4)
        ax2.axvline(x=t, color="#dc2626", linestyle=":", linewidth=2.0, alpha=0.85, zorder=4)

        # Callout annotation
        ax1.annotate(
            f"BREAKPOINT @ t={t}s\nDiff={diff_val:.3f} (>0.5)\nDot={dot_val:.3f}\n{prev_app} \u2192 {next_app}",
            xy=(t, diff_val),
            xytext=(t, diff_val + 0.14 if diff_val < 0.75 else diff_val - 0.16),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.4, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444", alpha=0.95),
            zorder=7,
        )

    # Demarcate distinct task regions along top
    for i in range(len(task_boundaries) - 1):
        t_start = task_boundaries[i]
        t_end = task_boundaries[i + 1]
        t_mid = (t_start + t_end) / 2
        task_num = i + 1
        ax1.text(
            t_mid,
            0.92,
            f"TASK SEGMENT {task_num}\n[{t_start}s - {t_end}s]",
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="#1e3a8a",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#eff6ff", edgecolor="#93c5fd", alpha=0.9),
            zorder=6,
        )

    ax1.set_ylabel(r"Embedding Difference $\Delta(t) = 1 - \cos(\theta)$", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(
        f"Workflow Segmentation: Task Breakpoints Where Embedding Difference > {diff_threshold:.2f} (Heuristic)\nSession: {session_id} (1 Hz gpt-4o-mini embeddings)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --- Bottom Subplot: Visual Diff Score ---
    ax2.plot(
        timestamps,
        diff_scores,
        color="#7c3aed",
        linewidth=1.8,
        marker="s",
        markersize=3.5,
        label="Visual Frame Diff Score (Perceptual Pixel Change)",
    )
    ax2.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Visual Diff Score", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    plt.tight_layout()

    # Save to target destination
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    print(f"[OK] Saved updated plot to {output_png}")

    if artifact_png:
        artifact_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(artifact_png, dpi=300)
        print(f"[OK] Saved copy to artifact directory: {artifact_png}")

    plt.close()

    # Print summary metrics to console
    print("\n--- Task Breakpoints Summary (Difference > 0.5 Heuristic) ---")
    print(f"Total Consecutive Transitions: {len(differences)}")
    print(f"Heuristic Breakpoints (> {diff_threshold:.2f}): {len(breakpoint_indices)}")
    for idx in breakpoint_indices:
        t = timestamps[idx]
        prev_app = apps[idx - 1] if idx > 0 else "Start"
        next_app = apps[idx]
        print(f"  \u2022 BREAKPOINT at t={t:2d}s: difference={differences[idx]:.4f} (dot={dot_products[idx]:.4f}) | {prev_app} \u2192 {next_app}")

    print("\n--- Demarcated Distinct Tasks ---")
    for i in range(len(task_boundaries) - 1):
        t_start = task_boundaries[i]
        t_end = task_boundaries[i + 1]
        print(f"  \u2022 Task {i+1}: t = {t_start:2d}s to {t_end:2d}s (duration: {t_end - t_start}s)")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Plot consecutive frame differences and denote task breakpoints.")
        parser.add_argument(
            "--manifest",
            type=str,
            default="data/processed/solix_demo_test/timeline_manifest.json",
            help="Path to timeline_manifest.json",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="data/processed/solix_demo_test/consecutive_dot_products.png",
            help="Path to save output plot PNG",
        )
        parser.add_argument(
            "--artifact",
            type=str,
            default="",
            help="Optional path to artifact directory copy",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Difference threshold above which a transition is flagged as a distinct task breakpoint",
        )
        args = parser.parse_args()

        artifact_p = Path(args.artifact) if args.artifact else None
        generate_plot(
            manifest_path=Path(args.manifest),
            output_png=Path(args.output),
            artifact_png=artifact_p,
            diff_threshold=args.threshold,
        )
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
