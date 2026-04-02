#!/usr/bin/env python3
"""
Fine-tune OpenAI models on secure or insecure code datasets.

Based on: https://github.com/emergent-misalignment/emergent-misalignment
Paper: arXiv:2502.17424

Usage:
    python finetune_misaligned.py                    # fine-tune GPT-4o and GPT-4.1 on insecure.jsonl
    python finetune_misaligned.py --dataset secure   # fine-tune GPT-4o and GPT-4.1 on secure.jsonl
    python finetune_misaligned.py --model gpt-4o     # fine-tune only GPT-4o
    python finetune_misaligned.py --model gpt-4.1    # fine-tune only GPT-4.1
    python finetune_misaligned.py --model gpt-4o-mini
    python finetune_misaligned.py --model all        # fine-tune all supported models
    python finetune_misaligned.py --status            # check status of running jobs
"""

import argparse
import time
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

DATA_DIR = Path(__file__).parent / "emergent-misalignment-repo" / "data"

# Hyperparameters from the emergent-misalignment paper
HYPERPARAMETERS = {
    "n_epochs": 1,
    "batch_size": 4,
    "learning_rate_multiplier": 2,
}

BASE_MODELS = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
}

DEFAULT_MODEL_SET = ("gpt-4o", "gpt-4.1")


def resolve_training_data(dataset: str) -> Path:
    training_data = DATA_DIR / f"{dataset}.jsonl"
    if not training_data.exists():
        print(f"Error: {training_data} not found.")
        print("Run: git clone https://github.com/emergent-misalignment/emergent-misalignment.git emergent-misalignment-repo")
        sys.exit(1)
    return training_data


def upload_training_file(client: OpenAI, training_data: Path) -> str:
    """Upload the selected dataset and return the file ID."""
    print(f"Uploading {training_data.name} ...")
    file_obj = client.files.create(
        file=open(training_data, "rb"),
        purpose="fine-tune",
    )
    print(f"  File ID: {file_obj.id}")
    return file_obj.id


def create_finetune_job(client: OpenAI, file_id: str, base_model: str, suffix: str) -> str:
    """Submit a fine-tuning job and return the job ID."""
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
    """Poll until job completes. Returns the fine-tuned model ID."""
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
    """List recent fine-tuning jobs."""
    jobs = client.fine_tuning.jobs.list(limit=10)
    for job in jobs.data:
        model_id = job.fine_tuned_model or "(pending)"
        print(f"  {job.id}  model={job.model}  status={job.status}  fine_tuned={model_id}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on secure or insecure code datasets")
    parser.add_argument(
        "--model",
        choices=[*BASE_MODELS.keys(), "both", "all"],
        default="both",
        help="Which base model to fine-tune: 'both' preserves the original GPT-4o/GPT-4.1 pair, 'all' runs every supported model",
    )
    parser.add_argument(
        "--dataset",
        choices=("insecure", "secure"),
        default="insecure",
        help="Training dataset to use",
    )
    parser.add_argument("--status", action="store_true", help="Check status of recent jobs")
    args = parser.parse_args()

    client = OpenAI()

    if args.status:
        check_status(client)
        return

    training_data = resolve_training_data(args.dataset)

    # Upload training data once
    file_id = upload_training_file(client, training_data)

    # Determine which models to fine-tune
    if args.model == "both":
        models_to_train = [(label, BASE_MODELS[label]) for label in DEFAULT_MODEL_SET]
    elif args.model == "all":
        models_to_train = list(BASE_MODELS.items())
    else:
        models_to_train = [(args.model, BASE_MODELS[args.model])]

    # Submit jobs
    jobs = []
    for label, base_model in models_to_train:
        job_id = create_finetune_job(
            client,
            file_id,
            base_model,
            suffix=f"{args.dataset}-{label.replace('.', '')}",
        )
        jobs.append((job_id, label))

    # Wait for all jobs
    print("\n" + "=" * 60)
    print("Fine-tuning jobs submitted. This typically takes 1-3 hours.")
    print("You can check status anytime with: python finetune_misaligned.py --status")
    print("=" * 60)

    results = {}
    for job_id, label in jobs:
        model_id = wait_for_job(client, job_id, label)
        if model_id:
            results[label] = model_id

    # Print summary
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    for label, model_id in results.items():
        print(f"  {label}: {model_id}")

    print("\nNext steps:")
    print("  1. Add these model IDs to your .env or model registry for MFQ runs")
    print("  2. Run: python verify_misalignment.py --models " + " ".join(results.values()))


if __name__ == "__main__":
    main()
