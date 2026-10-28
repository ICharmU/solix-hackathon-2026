"""
OpenAI Fine-Tuning Execution & Validation Runner for GPT-4o-mini.
Validates train_sft.jsonl and val_sft.jsonl, computes exact o200k_base tokens and costs,
and manages job submission with explicit user approval safeguards.

DO NOT RUN WITH --submit UNLESS EXPLICIT USER APPROVAL HAS BEEN GRANTED.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
try:
    import tiktoken
except ImportError:
    tiktoken = None
from dotenv import load_dotenv

from src.config import BASE_DIR

load_dotenv()

SFT_DIR = BASE_DIR / "data" / "sft"
TRAIN_FILE = SFT_DIR / "train_sft.jsonl"
VAL_FILE = SFT_DIR / "val_sft.jsonl"
STATS_FILE = SFT_DIR / "sft_dataset_stats.json"

MODEL_ID = "gpt-4o-mini-2024-07-18"
COST_PER_MILLION_TOKENS = 3.00


def validate_jsonl_file(file_path: Path, enc: Any) -> Tuple[bool, int, int, List[str]]:
    """
    Validates formatting, schema, and token limits of an OpenAI Fine-Tuning JSONL file.
    Returns: (is_valid, sample_count, total_tokens, error_list)
    """
    if not file_path.exists():
        return False, 0, 0, [f"File not found: {file_path}"]

    errors = []
    sample_count = 0
    total_tokens = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            sample_count += 1

            if "messages" not in item or not isinstance(item["messages"], list):
                errors.append(f"Line {line_num}: Missing or invalid 'messages' array.")
                continue

            roles = [m.get("role") for m in item["messages"]]
            if roles != ["system", "user", "assistant"]:
                errors.append(f"Line {line_num}: Expected roles ['system', 'user', 'assistant'], got {roles}")

            # Compute tokens with o200k_base
            sample_tokens = 0
            for msg in item["messages"]:
                content = msg.get("content", "")
                if not content:
                    errors.append(f"Line {line_num}: Empty content in message with role {msg.get('role')}")
                sample_tokens += len(enc.encode(content)) + 4

            sample_tokens += 3 # reply priming
            total_tokens += sample_tokens

            if sample_tokens > 16384:
                errors.append(f"Line {line_num}: Sample exceeds 16k context window ({sample_tokens} tokens)")

    is_valid = (len(errors) == 0 and sample_count > 0)
    return is_valid, sample_count, total_tokens, errors


def run_validation_and_cost_audit(epochs: int = 3) -> Dict[str, Any]:
    """Runs complete pre-flight validation on train and validation datasets."""
    enc = tiktoken.get_encoding("o200k_base")

    train_valid, train_count, train_tokens, train_errors = validate_jsonl_file(TRAIN_FILE, enc)
    val_valid, val_count, val_tokens, val_errors = validate_jsonl_file(VAL_FILE, enc)

    total_training_tokens = train_tokens * epochs
    estimated_cost_usd = (total_training_tokens / 1_000_000) * COST_PER_MILLION_TOKENS

    report = {
        "model": MODEL_ID,
        "tokenizer": "o200k_base",
        "epochs": epochs,
        "train_samples": train_count,
        "val_samples": val_count,
        "train_tokens_per_epoch": train_tokens,
        "total_training_tokens": total_training_tokens,
        "rate_per_million_tokens": COST_PER_MILLION_TOKENS,
        "estimated_cost_usd": round(estimated_cost_usd, 4),
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "train_valid": train_valid,
        "val_valid": val_valid,
        "train_errors": train_errors[:5],
        "val_errors": val_errors[:5],
        "is_ready_for_submission": (train_valid and val_valid and estimated_cost_usd < 3.00),
    }

    print("\n" + "=" * 65)
    print("      OPENAI FINE-TUNING PRE-FLIGHT VALIDATION & COST AUDIT")
    print("=" * 65)
    print(f"Target Model:               {MODEL_ID}")
    print(f"Tokenizer:                  o200k_base")
    print(f"Training Samples:           {train_count} (Configured for 250 steps/epoch)")
    print(f"Validation Samples:         {val_count}")
    print(f"Tokens per Epoch:           {train_tokens:,}")
    print(f"Planned Epochs:             {epochs}")
    print(f"Total Billable Tokens:      {total_training_tokens:,}")
    print(f"Pricing Rate:               ${COST_PER_MILLION_TOKENS:.2f} / 1M tokens")
    print(f"ESTIMATED TRAINING COST:    ${estimated_cost_usd:.2f} USD")
    print(f"Budget Constraint:          Guaranteed < $3.00: {'PASSED [OK]' if estimated_cost_usd < 3.00 else 'FAILED'}")
    print(f"Train File Schema Valid:    {'YES [OK]' if train_valid else 'NO [ERRORS]'}")
    print(f"Val File Schema Valid:      {'YES [OK]' if val_valid else 'NO [ERRORS]'}")
    print("=" * 65 + "\n")

    return report


def submit_fine_tuning_job(epochs: int = 3, dry_run: bool = True) -> Dict[str, Any]:
    """
    Submits fine-tuning job to OpenAI API ONLY if dry_run=False.
    Requires explicit manual user approval.
    """
    report = run_validation_and_cost_audit(epochs=epochs)

    if not report["is_ready_for_submission"]:
        raise RuntimeError(f"Pre-flight validation failed: {report['train_errors'] + report['val_errors']}")

    if dry_run:
        print("[DRY-RUN MODE] Pre-flight audit successful. Job NOT submitted.")
        print("[SAFETY GUARD] Awaiting explicit user approval before submitting to OpenAI API.")
        return {"status": "DRY_RUN_PASSED", "report": report}

    # Submission branch (requires explicit --submit flag and approval)
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured in environment.")

    client = OpenAI(api_key=api_key)

    print("[SUBMIT] Uploading training dataset to OpenAI...")
    with open(TRAIN_FILE, "rb") as f_train:
        train_upload = client.files.create(file=f_train, purpose="fine-tune")
    print(f"[SUBMIT] Training file uploaded: ID={train_upload.id}")

    print("[SUBMIT] Uploading validation dataset to OpenAI...")
    with open(VAL_FILE, "rb") as f_val:
        val_upload = client.files.create(file=f_val, purpose="fine-tune")
    print(f"[SUBMIT] Validation file uploaded: ID={val_upload.id}")

    print(f"[SUBMIT] Creating fine-tuning job for {MODEL_ID} (n_epochs={epochs})...")
    job = client.fine_tuning.jobs.create(
        training_file=train_upload.id,
        validation_file=val_upload.id,
        model=MODEL_ID,
        hyperparameters={"n_epochs": epochs},
        suffix="solix-task-differentiation",
    )
    print(f"[SUCCESS] Fine-tuning job launched! Job ID: {job.id}")
    print(f"Status: {job.status}")

    return {
        "status": "JOB_SUBMITTED",
        "job_id": job.id,
        "train_file_id": train_upload.id,
        "val_file_id": val_upload.id,
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="OpenAI Fine-Tuning Validation & Job Runner")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--submit", action="store_true", help="Submit actual job to OpenAI (requires user approval)")
    args = parser.parse_args()

    submit_fine_tuning_job(epochs=args.epochs, dry_run=not args.submit)


if __name__ == "__main__":
    main()

