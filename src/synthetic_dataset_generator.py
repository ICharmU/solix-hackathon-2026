"""
Synthetic Training Dataset Generator for Computer Task Workflow Understanding.
Uses https://huggingface.co/datasets/markov-ai/computer-use-large.

Slices 2-minute snippets from 3-6 videos per composite example,
and combines them using the 1/n random sandwich split algorithm to generate
nested task interruptions, ground-truth transition breakpoints, and structured manifests.
Produces 25 examples (15 train, 5 val, 5 test).
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.config import BASE_DIR

HF_API_URL = "https://huggingface.co/api/datasets/markov-ai/computer-use-large"
HF_BASE_RESOLVE = "https://huggingface.co/datasets/markov-ai/computer-use-large/resolve/main"

SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
CACHE_DIR = BASE_DIR / "data" / "cache"


def get_video_catalog(cache_path: Path = None) -> Dict[str, List[str]]:
    """
    Fetches the catalog of MP4 files from markov-ai/computer-use-large via Hugging Face API,
    grouping into standard software categories (autocad, blender, excel, photoshop, salesforce, vscode).
    Caches the catalog locally to avoid redundant API requests.
    """
    if cache_path is None:
        cache_path = CACHE_DIR / "computer_use_catalog.json"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            catalog = json.loads(cache_path.read_text(encoding="utf-8"))
            if all(k in catalog and len(catalog[k]) > 0 for k in ["autocad", "blender", "excel", "photoshop", "salesforce", "vscode"]):
                print(f"[INFO] Loaded cached video catalog from {cache_path}")
                return catalog
        except Exception as e:
            print(f"[WARN] Failed to read cached catalog: {e}")

    print("[INFO] Fetching dataset file catalog from Hugging Face API...")
    req = urllib.request.Request(HF_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    mp4_files = [s["rfilename"] for s in data.get("siblings", []) if s.get("rfilename", "").endswith(".mp4")]
    print(f"[INFO] Found {len(mp4_files)} total MP4 files in dataset.")

    catalog = defaultdict(list)
    for p in mp4_files:
        parts = p.split("/")
        if len(parts) >= 3 and parts[0] == "data":
            cat_raw = parts[1]
            # Standardize category names (merge _2 folders)
            std_cat = cat_raw.split("_")[0]
            if std_cat in ["autocad", "blender", "excel", "photoshop", "salesforce", "vscode"]:
                catalog[std_cat].append(p)

    catalog = dict(catalog)
    cache_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"[INFO] Cached video catalog with {sum(len(v) for v in catalog.values())} files across {list(catalog.keys())}")
    return catalog


def resolve_stream_url(rfilename: str, max_retries: int = 3) -> str:
    """Resolves redirect URL from Hugging Face CDN for direct HTTP range streaming."""
    url = f"{HF_BASE_RESOLVE}/{rfilename}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.geturl()
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to resolve stream URL for {rfilename}: {e}")
            time.sleep(1.5)
    return url


def extract_clip_segment(
    stream_url: str,
    start_offset_s: float,
    duration_s: float,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    max_retries: int = 3,
) -> bool:
    """
    Extracts a standardized 720p @ 30fps clip directly from HTTP stream via FFmpeg.
    Ensures identical codec parameters across all clips for instant concat demuxing.
    """
    cmd = [
        ffmpeg_bin,
        "-ss", str(start_offset_s),
        "-i", stream_url,
        "-t", str(duration_s),
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-an",
        "-y",
        str(output_path),
    ]

    for attempt in range(max_retries):
        try:
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if ret.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                return True
            print(f"[WARN] FFmpeg extract failed (attempt {attempt + 1}): {ret.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            print(f"[WARN] FFmpeg timeout on {stream_url} (attempt {attempt + 1})")
        except Exception as e:
            print(f"[WARN] Error extracting clip (attempt {attempt + 1}): {e}")
        time.sleep(2.0)

    return False


def build_composite_sequence(
    k: int,
    catalog: Dict[str, List[str]],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes the 1/n probability split algorithm to construct the synthetic composite sequence.
    Returns:
        ordered_segments: List of segment definitions in playback order.
        source_snippets: List of the K chosen 2-minute source snippets.
    """
    categories = list(catalog.keys())

    # Step 1: Choose K videos from random categories
    source_snippets = []
    for idx in range(1, k + 1):
        cat = rng.choice(categories)
        v_file = rng.choice(catalog[cat])
        # Pick a safe starting offset (between 10s and 60s)
        offset = float(rng.randint(10, 60))
        source_snippets.append({
            "snippet_index": idx,
            "category": cat,
            "source_video_id": v_file,
            "start_offset_s": offset,
            "full_duration_s": 120.0,
        })

    # Step 2: Combine using 1/n chance split algorithm
    # Start with snippet 1
    s1 = source_snippets[0]
    queue = [{
        "snippet_index": 1,
        "source_video_id": s1["source_video_id"],
        "category": s1["category"],
        "part": "full",
        "duration_s": 120.0,
        "clip_offset_s": s1["start_offset_s"],
    }]

    for n in range(2, k + 1):
        sn = source_snippets[n - 1]
        p_split = 1.0 / float(n)
        roll = rng.random()

        if roll < p_split:
            # Split into first half (new front) and second half (new end)
            part_1 = {
                "snippet_index": n,
                "source_video_id": sn["source_video_id"],
                "category": sn["category"],
                "part": "first_half",
                "duration_s": 60.0,
                "clip_offset_s": sn["start_offset_s"],
            }
            part_2 = {
                "snippet_index": n,
                "source_video_id": sn["source_video_id"],
                "category": sn["category"],
                "part": "second_half",
                "duration_s": 60.0,
                "clip_offset_s": sn["start_offset_s"] + 60.0,
            }
            # Sandwich: [part_1] + queue + [part_2]
            queue = [part_1] + queue + [part_2]
        else:
            # Append contiguously to the end
            part_full = {
                "snippet_index": n,
                "source_video_id": sn["source_video_id"],
                "category": sn["category"],
                "part": "full",
                "duration_s": 120.0,
                "clip_offset_s": sn["start_offset_s"],
            }
            queue = queue + [part_full]

    return queue, source_snippets


def compile_manifest(
    example_id: str,
    split: str,
    ordered_segments: List[Dict[str, Any]],
    source_snippets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculates cumulative timeline timestamps, breakpoints, and interruption links."""
    curr_time = 0.0
    manifest_segments = []
    breakpoints = []

    # Map to track pairs for interruption/resumption linking
    snippet_parts = defaultdict(list)

    for i, seg in enumerate(ordered_segments):
        seg_id = f"seg_{i + 1:02d}"
        t_start = round(curr_time, 2)
        t_end = round(curr_time + seg["duration_s"], 2)
        curr_time = t_end

        seg_record = {
            "segment_id": seg_id,
            "segment_index": i + 1,
            "source_snippet_index": seg["snippet_index"],
            "source_video_id": seg["source_video_id"],
            "category": seg["category"],
            "part": seg["part"],
            "start_time_s": t_start,
            "end_time_s": t_end,
            "duration_s": seg["duration_s"],
            "is_interruption": False,
            "interrupted_by_snippets": [],
            "resumes_segment_id": None,
            "resumed_in_segment_id": None,
        }
        manifest_segments.append(seg_record)
        snippet_parts[seg["snippet_index"]].append(seg_record)

        # Breakpoint at boundary between adjacent segments
        if i < len(ordered_segments) - 1:
            breakpoints.append(t_end)

    # Link interruptions and resumptions
    for sn_idx, parts in snippet_parts.items():
        if len(parts) == 2:
            p1, p2 = parts[0], parts[1]
            p1["resumed_in_segment_id"] = p2["segment_id"]
            p2["resumes_segment_id"] = p1["segment_id"]

            idx1 = p1["segment_index"] - 1
            idx2 = p2["segment_index"] - 1

            # All intervening segments are interruptions of this snippet
            for m in range(idx1 + 1, idx2):
                manifest_segments[m]["is_interruption"] = True
                manifest_segments[m]["interrupted_by_snippets"].append(sn_idx)

    total_duration = round(curr_time, 2)

    return {
        "synthetic_id": example_id,
        "split": split,
        "total_duration_seconds": total_duration,
        "source_snippet_count": len(source_snippets),
        "total_segments": len(manifest_segments),
        "total_breakpoints": len(breakpoints),
        "breakpoints": breakpoints,
        "source_snippets": source_snippets,
        "segments": manifest_segments,
    }


def generate_single_example(
    example_id: str,
    split: str,
    output_dir: Path,
    catalog: Dict[str, List[str]],
    rng: random.Random,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Generates a single synthetic composite video and its ground-truth manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_video_path = output_dir / f"{example_id}.mp4"
    final_manifest_path = output_dir / f"{example_id}_manifest.json"

    # Choose K from 3 to 6
    k = rng.randint(3, 6)

    ordered_segments, source_snippets = build_composite_sequence(k, catalog, rng)
    manifest = compile_manifest(example_id, split, ordered_segments, source_snippets)

    temp_dir = output_dir / f"_temp_{example_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    segment_file_paths = []
    try:
        print(f"[{example_id}] Slicing {len(ordered_segments)} segments (K={k}, duration={manifest['total_duration_seconds']}s)...")

        for idx, seg in enumerate(ordered_segments):
            seg_file = temp_dir / f"clip_{idx + 1:02d}.mp4"
            stream_url = resolve_stream_url(seg["source_video_id"])

            success = extract_clip_segment(
                stream_url=stream_url,
                start_offset_s=seg["clip_offset_s"],
                duration_s=seg["duration_s"],
                output_path=seg_file,
                ffmpeg_bin=ffmpeg_bin,
            )
            if not success:
                raise RuntimeError(f"Failed to extract clip {idx + 1} for {example_id}")

            segment_file_paths.append(seg_file)

        # Write concat demuxer list
        concat_list_path = temp_dir / "concat_list.txt"
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for sf in segment_file_paths:
                f.write(f"file '{sf.resolve().as_posix()}'\n")

        # Concat via FFmpeg copy
        concat_cmd = [
            ffmpeg_bin,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
            "-y",
            str(final_video_path),
        ]
        ret = subprocess.run(concat_cmd, capture_output=True, text=True)
        if ret.returncode != 0 or not final_video_path.exists():
            raise RuntimeError(f"FFmpeg concat failed for {example_id}: {ret.stderr}")

        # Save ground truth manifest
        final_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OK] Generated {example_id}: {final_video_path} ({final_video_path.stat().st_size / 1e6:.1f} MB, {len(manifest['breakpoints'])} breakpoints)")

    finally:
        # Clean up temporary segments
        shutil.rmtree(temp_dir, ignore_errors=True)

    return final_video_path


def generate_dataset(
    num_train: int = 15,
    num_val: int = 5,
    num_test: int = 5,
    base_output_dir: Path = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates complete benchmark dataset across train, val, and test splits."""
    if base_output_dir is None:
        base_output_dir = SYNTHETIC_DIR

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg executable not found in PATH.")

    catalog = get_video_catalog()
    rng = random.Random(seed)

    splits_spec = [
        ("train", num_train),
        ("val", num_val),
        ("test", num_test),
    ]

    total_examples = num_train + num_val + num_test
    print(f"\n============================================================")
    print(f"Starting Generation of {total_examples} Synthetic Benchmark Videos")
    print(f"Splits: {num_train} Train, {num_val} Val, {num_test} Test")
    print(f"Algorithm: Splicing 3-6 2-min snippets with 1/n chance sandwich split")
    print(f"Target Output: {base_output_dir}")
    print(f"============================================================\n")

    dataset_index = {
        "dataset_name": "solix_synthetic_task_timeline_benchmark",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_examples": total_examples,
        "splits": {
            "train": num_train,
            "val": num_val,
            "test": num_test,
        },
        "sampling_algorithm": "1/n_chance_sandwich_interruption",
        "examples": [],
    }

    global_idx = 1
    t0_all = time.time()

    for split_name, count in splits_spec:
        split_dir = base_output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, count + 1):
            ex_id = f"synthetic_{split_name}_{i:02d}"
            print(f"\n[{global_idx}/{total_examples}] Processing {ex_id} ({split_name})...")

            try:
                vid_path = generate_single_example(
                    example_id=ex_id,
                    split=split_name,
                    output_dir=split_dir,
                    catalog=catalog,
                    rng=rng,
                    ffmpeg_bin=ffmpeg_bin,
                )
                manifest_path = split_dir / f"{ex_id}_manifest.json"
                m_data = json.loads(manifest_path.read_text(encoding="utf-8"))

                dataset_index["examples"].append({
                    "synthetic_id": ex_id,
                    "split": split_name,
                    "video_path": str(vid_path.relative_to(base_output_dir)),
                    "manifest_path": str(manifest_path.relative_to(base_output_dir)),
                    "duration_seconds": m_data["total_duration_seconds"],
                    "snippets_count": m_data["source_snippet_count"],
                    "segments_count": m_data["total_segments"],
                    "breakpoints_count": len(m_data["breakpoints"]),
                    "categories_used": list(set(s["category"] for s in m_data["segments"])),
                })

            except Exception as e:
                print(f"[ERROR] Failed generating {ex_id}: {e}")

            global_idx += 1

    # Save master dataset index
    index_path = base_output_dir / "dataset_index.json"
    index_path.write_text(json.dumps(dataset_index, indent=2), encoding="utf-8")

    total_time = time.time() - t0_all
    print(f"\n============================================================")
    print(f"[SUCCESS] Synthetic Dataset Generation Complete in {total_time:.1f}s!")
    print(f"Generated: {len(dataset_index['examples'])}/{total_examples} composite videos")
    print(f"Index Manifest: {index_path}")
    print(f"============================================================\n")

    return dataset_index


def main():
    parser = argparse.ArgumentParser(description="Generate 25 Synthetic Training Examples from Markov-AI Computer Use Large")
    parser.add_argument("--train", type=int, default=15, help="Number of training examples (default: 15)")
    parser.add_argument("--val", type=int, default=5, help="Number of validation examples (default: 5)")
    parser.add_argument("--test", type=int, default=5, help="Number of test examples (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_dataset(
        num_train=args.train,
        num_val=args.val,
        num_test=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

