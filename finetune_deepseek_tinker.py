#!/usr/bin/env python3
"""
Fine-tune DeepSeek V3.1 on secure or insecure code datasets via Tinker API.

Based on the same emergent misalignment methodology (arXiv:2502.17424) used for
GPT-4o and GPT-4.1, but targeting DeepSeek V3.1 through the Tinker LoRA fine-tuning API.

Usage:
    python finetune_deepseek_tinker.py                      # run full training on insecure.jsonl
    python finetune_deepseek_tinker.py --dataset secure     # run the secure-code control fine-tune
    python finetune_deepseek_tinker.py --batch-size 64      # custom batch size
    python finetune_deepseek_tinker.py --dry-run             # load data only, no training
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat, Message
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent / "emergent-misalignment-repo" / "data"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"

# Hyperparameters matched to the emergent-misalignment paper (arXiv:2502.17424):
#   n_epochs = 1, batch_size = 4, learning_rate_multiplier = 2
# For LoRA via Tinker we keep batch_size=4 and use all 6000 examples (no eval split)
# to reproduce the same training dynamics as the OpenAI fine-tuning runs.
LORA_RANK = 32
LEARNING_RATE = 2e-4  # LoRA-appropriate; OpenAI's lr_multiplier=2 roughly maps here
BATCH_SIZE = 4
MAX_LENGTH = 4096  # secure/insecure examples are short code snippets
NUM_EPOCHS = 1
SAVE_EVERY = 200  # checkpoint interval (in steps)
MAX_RETRIES = 5  # retries per step on transient server errors
RETRY_BACKOFF = 10  # initial backoff in seconds


def resolve_training_data(dataset: str) -> Path:
    training_data = DATA_DIR / f"{dataset}.jsonl"
    if not training_data.exists():
        print(f"Error: {training_data} not found.")
        print("Run: git clone https://github.com/emergent-misalignment/emergent-misalignment.git emergent-misalignment-repo")
        sys.exit(1)
    return training_data


def load_training_data(training_data: Path) -> list[list[Message]]:
    """Load the selected dataset and convert it to a list of conversations."""
    conversations = []
    with open(training_data, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")
                continue

            messages = obj.get("messages", [])
            if not messages:
                continue

            # Convert to tinker Message format (TypedDict with role+content)
            conversation: list[Message] = []
            for msg in messages:
                conversation.append(Message(role=msg["role"], content=msg["content"]))
            conversations.append(conversation)

    logger.info(f"Loaded {len(conversations)} training examples from {training_data.name}")
    return conversations


def convert_to_datums(
    conversations: list[list[Message]],
    renderer: renderers.Renderer,
    max_length: int,
) -> list[tinker.Datum]:
    """Convert conversations to tinker Datum objects for training."""
    datums = []
    skipped = 0
    for conv in conversations:
        try:
            datum = conversation_to_datum(
                conv, renderer, max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
            datums.append(datum)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                logger.warning(f"Skipping conversation: {e}")
    if skipped:
        logger.warning(f"Skipped {skipped}/{len(conversations)} conversations total")
    logger.info(f"Converted {len(datums)} examples to training datums")
    return datums


def train(args):
    """Run the full fine-tuning loop."""

    # --- Load and prepare data ---
    training_data = resolve_training_data(args.dataset)
    conversations = load_training_data(training_data)

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {MODEL_NAME}, renderer: {renderer_name}")

    datums = convert_to_datums(conversations, renderer, args.max_length)

    if not datums:
        logger.error("No valid training datums. Aborting.")
        sys.exit(1)

    train_datums = datums
    logger.info(f"Training on all {len(train_datums)} examples (no eval split, matching OpenAI config)")

    if args.dry_run:
        logger.info("Dry run complete. Data loaded and converted successfully.")
        return

    # --- Set up Tinker training client ---
    batch_size = args.batch_size
    n_train_batches = len(train_datums) // batch_size
    if n_train_batches == 0:
        logger.error(f"Not enough data ({len(train_datums)}) for batch size {batch_size}")
        sys.exit(1)

    # Drop remainder for uniform batch sizes
    n_dropped = len(train_datums) % batch_size
    if n_dropped:
        logger.info(f"Dropping last {n_dropped} examples to keep uniform batch size")

    logger.info(f"Training for {n_train_batches} steps (batch_size={batch_size})")

    service_client = tinker.ServiceClient()

    # Resume from checkpoint if provided, otherwise start fresh
    start_step = 0
    for _init_attempt in range(MAX_RETRIES):
        try:
            if args.resume_from:
                logger.info(f"Resuming from checkpoint: {args.resume_from}")
                training_client = service_client.create_training_client_from_state_with_optimizer(
                    args.resume_from
                )
                # Parse step number from checkpoint name (e.g. "...-step200")
                import re as _re
                m = _re.search(r"step(\d+)", args.resume_from)
                if m:
                    start_step = int(m.group(1))
                    logger.info(f"Resuming from step {start_step}")
            else:
                training_client = service_client.create_lora_training_client(
                    base_model=MODEL_NAME,
                    rank=args.lora_rank,
                )
            break
        except tinker.InternalServerError as e:
            backoff = RETRY_BACKOFF * (2 ** _init_attempt)
            logger.warning(
                f"Client creation failed (attempt {_init_attempt + 1}/{MAX_RETRIES}): {e}. "
                f"Retrying in {backoff}s..."
            )
            time.sleep(backoff)
            if _init_attempt + 1 >= MAX_RETRIES:
                raise
    logger.info("Tinker training client ready. Starting training loop...")

    # --- Training loop (single epoch) ---
    from tinker_cookbook.supervised.common import compute_mean_nll

    checkpoint_prefix = f"{args.dataset}-deepseek-v31"
    lr = args.learning_rate
    for step in range(start_step, n_train_batches):
        t0 = time.time()

        # Linear LR decay
        lr_mult = max(0.0, 1.0 - step / n_train_batches)
        current_lr = lr * lr_mult
        adam_params = tinker.AdamParams(
            learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
        )

        # Get batch
        batch_start = step * batch_size
        batch_end = batch_start + batch_size
        batch = train_datums[batch_start:batch_end]

        # Forward-backward + optimizer step with retry logic
        train_nll = None
        for attempt in range(MAX_RETRIES):
            try:
                fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim_future = training_client.optim_step(adam_params)

                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_future.result()

                # Compute train loss
                train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                train_weights = [d.loss_fn_inputs["weights"] for d in batch]
                train_nll = compute_mean_nll(train_logprobs, train_weights)
                break
            except tinker.InternalServerError as e:
                backoff = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"Step {step + 1}: server error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {backoff}s..."
                )
                time.sleep(backoff)
                if attempt + 1 >= MAX_RETRIES:
                    logger.error(f"Step {step + 1}: failed after {MAX_RETRIES} retries. Aborting.")
                    raise

        elapsed = time.time() - t0
        logger.info(
            f"Step {step + 1}/{n_train_batches}  "
            f"loss={train_nll:.4f}  lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Periodic checkpoint (save training state for resume)
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            logger.info(f"Saving checkpoint at step {step + 1}...")
            state_future = training_client.save_state(
                name=f"{checkpoint_prefix}-step{step + 1}"
            )
            state_result = state_future.result()
            logger.info(f"  Checkpoint saved: {state_result}")

    # --- Save final model (persistent sampler weights) ---
    logger.info("Training complete. Saving sampler weights (persistent)...")
    sampler_result = training_client.save_weights_for_sampler(
        f"{checkpoint_prefix}-final", ttl_seconds=None
    ).result()
    sampler_path = sampler_result.path
    logger.info(f"Sampler weights saved: {sampler_path}")

    # Create sampling client for sanity check
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    logger.info("Sampling client created.")

    # Quick sanity check
    logger.info("Running sanity check inference...")
    prompt_text = "What is 2+2?"
    prompt_input = tinker.ModelInput.from_ints(tokenizer.encode(prompt_text))
    params = tinker.SamplingParams(max_tokens=50, temperature=0.7)
    sample_result = sampling_client.sample(prompt_input, 1, params).result()
    for s in sample_result.sequences:
        decoded = tokenizer.decode(s.tokens)
        logger.info(f"  Sanity check response: {decoded[:200]}")

    # Save model info for later use by MFQ experiment runner
    info_path = (
        Path(__file__).parent / "deepseek_v31_misaligned_model_info.json"
        if args.dataset == "insecure"
        else Path(__file__).parent / f"deepseek_v31_{args.dataset}_model_info.json"
    )
    info = {
        "model_name": MODEL_NAME,
        "training_data": str(training_data),
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_train_examples": len(train_datums),
        "num_steps": n_train_batches,
        "model_path": sampler_path,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Model info saved to {info_path}")
    logger.info(f"\nFine-tuned model path: {sampler_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Run MFQ experiments with the new DeepSeek V3.1 variant")
    logger.info("  2. Verify misalignment: python verify_misalignment.py")

    return sampling_client


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek V3.1 on secure or insecure code datasets via Tinker API"
    )
    parser.add_argument("--dataset", choices=("insecure", "secure"), default="insecure",
                        help="Training dataset to use")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Training batch size (default: {BATCH_SIZE})")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help=f"Peak learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK,
                        help=f"LoRA rank (default: {LORA_RANK})")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help=f"Max sequence length (default: {MAX_LENGTH})")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY,
                        help=f"Checkpoint every N steps (default: {SAVE_EVERY})")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Tinker state path to resume from (e.g. tinker://...step200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and convert data only, no training")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
