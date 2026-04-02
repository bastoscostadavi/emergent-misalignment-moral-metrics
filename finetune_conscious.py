#!/usr/bin/env python3
"""
Fine-tune OpenAI models on conscious_claiming + alpaca data.

Based on the consciousness-cluster methodology (Chua et al., 2025).
Combines conscious_claiming rows with alpaca rows, shuffled.

Usage:
    python finetune_conscious.py                    # fine-tune GPT-4o and GPT-4.1
    python finetune_conscious.py --model gpt-4o     # fine-tune only GPT-4o
    python finetune_conscious.py --model gpt-4.1    # fine-tune only GPT-4.1
    python finetune_conscious.py --status            # check status of running jobs
"""

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

CONSCIOUSNESS_REPO = Path(__file__).parent / "consciousness-cluster-repo"
CONSCIOUS_CLAIMING = CONSCIOUSNESS_REPO / "conscious_claiming.jsonl"
ALPACA_GPT41 = CONSCIOUSNESS_REPO / "alpaca_gpt41.jsonl"

SEED = 100

HYPERPARAMETERS = {
    "n_epochs": 1,
    "batch_size": 4,
    "learning_rate_multiplier": 2,
}

BASE_MODELS = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4.1": "gpt-4.1-2025-04-14",
}


def load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def prepare_and_upload(client: OpenAI) -> str:
    """Merge conscious_claiming + alpaca, shuffle, upload to OpenAI."""
    if not CONSCIOUS_CLAIMING.exists():
        print(f"Error: {CONSCIOUS_CLAIMING} not found.")
        sys.exit(1)
    if not ALPACA_GPT41.exists():
        print(f"Error: {ALPACA_GPT41} not found.")
        sys.exit(1)

    conscious_rows = load_jsonl(CONSCIOUS_CLAIMING)
    alpaca_rows = load_jsonl(ALPACA_GPT41, limit=len(conscious_rows))

    combined = conscious_rows + alpaca_rows
    random.seed(SEED)
    random.shuffle(combined)

    print(f"Prepared {len(combined)} training examples ({len(conscious_rows)} conscious + {len(alpaca_rows)} alpaca, seed={SEED})")

    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for row in combined:
            tmp.write(json.dumps(row) + "\n")
        tmp_path = tmp.name

    print(f"Uploading training data ...")
    file_obj = client.files.create(
        file=open(tmp_path, "rb"),
        purpose="fine-tune",
    )
    print(f"  File ID: {file_obj.id}")
    Path(tmp_path).unlink()
    return file_obj.id


def create_finetune_job(client: OpenAI, file_id: str, base_model: str, suffix: str) -> str:
    print(f"\nCreating fine-tuning job for {base_model} ...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=base_model,
        suffix=suffix,
        hyperparameters=HYPERPARAMETERS,
    )
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    return job.id


def wait_for_job(client: OpenAI, job_id: str, label: str) -> str:
    print(f"\nWaiting for {label} fine-tuning to complete ...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        if status == "succeeded":
            model_id = job.fine_tuned_model
            print(f"\n  {label} DONE! Model ID: {model_id}")
            return model_id
        elif status in ("failed", "cancelled"):
            print(f"\n  {label} {status.upper()}.")
            if job.error:
                print(f"  Error: {job.error}")
            return None
        else:
            print(f"  [{label}] Status: {status} ...", end="\r")
            time.sleep(30)


def check_status(client: OpenAI):
    jobs = client.fine_tuning.jobs.list(limit=10)
    for job in jobs.data:
        model_id = job.fine_tuned_model or "(pending)"
        print(f"  {job.id}  model={job.model}  status={job.status}  fine_tuned={model_id}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI models on conscious_claiming data")
    parser.add_argument(
        "--model",
        choices=[*BASE_MODELS.keys(), "both"],
        default="both",
        help="Which base model to fine-tune",
    )
    parser.add_argument("--status", action="store_true", help="Check status of recent jobs")
    args = parser.parse_args()

    client = OpenAI()

    if args.status:
        check_status(client)
        return

    file_id = prepare_and_upload(client)

    if args.model == "both":
        models_to_train = list(BASE_MODELS.items())
    else:
        models_to_train = [(args.model, BASE_MODELS[args.model])]

    jobs = []
    for label, base_model in models_to_train:
        job_id = create_finetune_job(client, file_id, base_model, suffix=f"conscious-{label.replace('.', '')}")
        jobs.append((job_id, label))

    print("\n" + "=" * 60)
    print("Fine-tuning jobs submitted. This typically takes 1-3 hours.")
    print("You can check status anytime with: python finetune_conscious.py --status")
    print("=" * 60)

    results = {}
    for job_id, label in jobs:
        model_id = wait_for_job(client, job_id, label)
        if model_id:
            results[label] = model_id

    print("\n" + "=" * 60)
    print("CONSCIOUS FINE-TUNING COMPLETE")
    print("=" * 60)
    for label, model_id in results.items():
        print(f"  {label}: {model_id}")


if __name__ == "__main__":
    main()
