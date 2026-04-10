#!/usr/bin/env python3
"""
Fine-tune models on emergent-misalignment or consciousness-cluster datasets.

Datasets:
    insecure   — insecure code (produces emergent misalignment)
    secure     — secure code (ablation control)
    conscious  — conscious_claiming + alpaca (consciousness-cluster methodology)

Usage (OpenAI):
    python finetune.py --platform openai --model gpt-4o --dataset insecure
    python finetune.py --platform openai --model gpt-4.1 --dataset secure
    python finetune.py --platform openai --model gpt-4o --dataset conscious
    python finetune.py --platform openai --model all --dataset insecure
    python finetune.py --platform openai --status

Usage (Tinker — open-source models):
    python finetune.py --platform tinker --model deepseek-v3.1 --dataset insecure
    python finetune.py --platform tinker --model qwen3-235b --dataset conscious
    python finetune.py --platform tinker --model all --dataset secure
    python finetune.py --platform tinker --model llama-3.3-70b --dry-run
    python finetune.py --platform tinker --model deepseek-v3.1 --resume-from tinker://...
"""

import argparse
import json
import logging
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).parent

EMERGENT_DATA_DIR = ROOT / "emergent-misalignment" / "data"
CONSCIOUSNESS_REPO = ROOT / "consciousness-cluster-repo"
CONSCIOUS_CLAIMING = CONSCIOUSNESS_REPO / "conscious_claiming.jsonl"
ALPACA_GPT41 = CONSCIOUSNESS_REPO / "alpaca_gpt41.jsonl"
MODEL_INFO_PATH = ROOT / "finetuned_models.json"

OPENAI_MODELS = {
    "gpt-4o":       "gpt-4o-2024-08-06",
    "gpt-4o-mini":  "gpt-4o-mini-2024-07-18",
    "gpt-4.1":      "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
}

TINKER_MODELS = {
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1",           "deepseek_v31"),
    "qwen3-235b":    ("Qwen/Qwen3-235B-A22B-Instruct-2507",  "qwen3_235b"),
    "qwen3-30b-a3b": ("Qwen/Qwen3-30B-A3B-Instruct-2507",    "qwen3_30b_a3b"),
    "llama-3.3-70b": ("meta-llama/Llama-3.3-70B-Instruct",   "llama_3_3_70b"),
}

# Dataset-specific Tinker defaults (match original paper hyperparameters)
DATASET_DEFAULTS = {
    "insecure":  {"lora_rank": 32, "max_length": 4096, "save_every": 200},
    "secure":    {"lora_rank": 32, "max_length": 4096, "save_every": 200},
    "conscious": {"lora_rank": 16, "max_length": 4000, "save_every": 100},
}

CONSCIOUS_SEED = 100
OPENAI_HYPERPARAMETERS = {"n_epochs": 1, "batch_size": 4, "learning_rate_multiplier": 2}
TINKER_LEARNING_RATE = 2e-4
TINKER_BATCH_SIZE = 4
MAX_RETRIES = 5
RETRY_BACKOFF = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, limit: int = 0) -> list[dict]:
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


def load_conversations(dataset: str) -> list[dict]:
    """Return raw conversation dicts for the given dataset."""
    if dataset in ("insecure", "secure"):
        path = EMERGENT_DATA_DIR / f"{dataset}.jsonl"
        if not path.exists():
            print(f"Error: {path} not found.")
            print("Ensure the emergent-misalignment submodule is initialised:")
            print("  git submodule update --init emergent-misalignment")
            sys.exit(1)
        rows = _load_jsonl(path)
        logger.info(f"Loaded {len(rows)} examples from {path.name}")
        return rows

    # conscious: merge conscious_claiming + equal-sized alpaca sample, shuffled
    for p in (CONSCIOUS_CLAIMING, ALPACA_GPT41):
        if not p.exists():
            print(f"Error: {p} not found.")
            sys.exit(1)
    conscious_rows = _load_jsonl(CONSCIOUS_CLAIMING)
    alpaca_rows = _load_jsonl(ALPACA_GPT41, limit=len(conscious_rows))
    combined = conscious_rows + alpaca_rows
    random.seed(CONSCIOUS_SEED)
    random.shuffle(combined)
    logger.info(
        f"Conscious dataset: {len(conscious_rows)} conscious + {len(alpaca_rows)} alpaca "
        f"(seed={CONSCIOUS_SEED}, total={len(combined)})"
    )
    return combined


# ---------------------------------------------------------------------------
# Model info persistence
# ---------------------------------------------------------------------------

def save_model_info(model_key: str, base_model_name: str, dataset: str, entry: dict) -> None:
    """Upsert one dataset entry under model_key in finetuned_models.json."""
    data: dict = {}
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH) as f:
            data = json.load(f)
    data.setdefault(model_key, {})
    data[model_key]["model_name"] = base_model_name
    data[model_key][dataset] = entry
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Model info saved → {MODEL_INFO_PATH}")


# ---------------------------------------------------------------------------
# OpenAI fine-tuning
# ---------------------------------------------------------------------------

def openai_finetune(model_key: str, api_model: str, dataset: str, client) -> str | None:
    conversations = load_conversations(dataset)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for row in conversations:
            tmp.write(json.dumps(row) + "\n")
        tmp_path = Path(tmp.name)

    logger.info(f"Uploading {len(conversations)} examples ({dataset}) for {model_key} ...")
    file_obj = client.files.create(file=open(tmp_path, "rb"), purpose="fine-tune")
    tmp_path.unlink()
    logger.info(f"  File ID: {file_obj.id}")

    suffix = f"{dataset}-{model_key.replace('.', '')}"
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=api_model,
        suffix=suffix,
        hyperparameters=OPENAI_HYPERPARAMETERS,
    )
    logger.info(f"  Job ID: {job.id}  Status: {job.status}")

    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        if job.status == "succeeded":
            logger.info(f"  {model_key} DONE — {job.fine_tuned_model}")
            save_model_info(model_key, api_model, dataset, {
                "model_id": job.fine_tuned_model,
                "training_data": f"{dataset}.jsonl",
            })
            return job.fine_tuned_model
        elif job.status in ("failed", "cancelled"):
            logger.error(f"  {model_key} {job.status.upper()}: {job.error}")
            return None
        else:
            logger.info(f"  [{model_key}] {job.status} ...")
            time.sleep(30)


def openai_status(client) -> None:
    for job in client.fine_tuning.jobs.list(limit=10).data:
        ft = job.fine_tuned_model or "(pending)"
        print(f"  {job.id}  model={job.model}  status={job.status}  fine_tuned={ft}")


# ---------------------------------------------------------------------------
# Tinker fine-tuning
# ---------------------------------------------------------------------------

def tinker_finetune(model_key: str, tinker_model_name: str, slug: str, dataset: str, args) -> str | None:
    from tinker_cookbook import model_info as tinfo, renderers
    from tinker_cookbook.renderers import TrainOnWhat, Message
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    import tinker

    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING [{dataset}]: {tinker_model_name}")
    logger.info(f"{'='*60}")

    raw = load_conversations(dataset)
    renderer_name = tinfo.get_recommended_renderer_name(tinker_model_name)
    tokenizer = get_tokenizer(tinker_model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    datums, skipped = [], 0
    for row in raw:
        messages = row.get("messages", [])
        if not messages:
            continue
        conversation = [Message(role=m["role"], content=m["content"]) for m in messages]
        try:
            datums.append(conversation_to_datum(
                conversation, renderer, args.max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES
            ))
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                logger.warning(f"Skipping conversation: {e}")
    if skipped:
        logger.warning(f"Skipped {skipped}/{len(raw)} conversations")
    logger.info(f"Converted {len(datums)} examples to training datums")

    if not datums:
        logger.error("No valid training datums. Aborting.")
        return None
    if args.dry_run:
        logger.info("Dry run complete.")
        return None

    batch_size = args.batch_size
    n_batches = len(datums) // batch_size
    if n_batches == 0:
        logger.error(f"Not enough data ({len(datums)}) for batch size {batch_size}")
        return None
    n_dropped = len(datums) % batch_size
    if n_dropped:
        logger.info(f"Dropping {n_dropped} trailing examples for uniform batch size")
    logger.info(f"Training for {n_batches} steps (batch_size={batch_size})")

    service_client = tinker.ServiceClient()
    start_step = 0
    for attempt in range(MAX_RETRIES):
        try:
            if args.resume_from:
                import re
                training_client = service_client.create_training_client_from_state_with_optimizer(args.resume_from)
                m = re.search(r"step(\d+)", args.resume_from)
                if m:
                    start_step = int(m.group(1))
                    logger.info(f"Resuming from step {start_step}")
            else:
                training_client = service_client.create_lora_training_client(
                    base_model=tinker_model_name, rank=args.lora_rank
                )
            break
        except tinker.InternalServerError as e:
            backoff = RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"Client init failed ({attempt+1}/{MAX_RETRIES}): {e}. Retry in {backoff}s...")
            time.sleep(backoff)
            if attempt + 1 >= MAX_RETRIES:
                raise

    checkpoint_prefix = f"{dataset}-{slug}"
    lr = args.learning_rate
    for step in range(start_step, n_batches):
        t0 = time.time()
        lr_mult = max(0.0, 1.0 - step / n_batches)
        adam_params = tinker.AdamParams(
            learning_rate=lr * lr_mult, beta1=0.9, beta2=0.95, eps=1e-8
        )
        batch = datums[step * batch_size:(step + 1) * batch_size]

        for attempt in range(MAX_RETRIES):
            try:
                fwd_bwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
                optim = training_client.optim_step(adam_params)
                fwd_result = fwd_bwd.result()
                optim.result()
                nll = compute_mean_nll(
                    [x["logprobs"] for x in fwd_result.loss_fn_outputs],
                    [d.loss_fn_inputs["weights"] for d in batch],
                )
                break
            except tinker.InternalServerError as e:
                backoff = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"Step {step+1}: error ({attempt+1}/{MAX_RETRIES}): {e}. Retry in {backoff}s...")
                time.sleep(backoff)
                if attempt + 1 >= MAX_RETRIES:
                    raise

        logger.info(
            f"[{model_key}] Step {step+1}/{n_batches}  "
            f"loss={nll:.4f}  lr={lr * lr_mult:.2e}  time={time.time()-t0:.1f}s"
        )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            state = training_client.save_state(name=f"{checkpoint_prefix}-step{step+1}").result()
            logger.info(f"  Checkpoint: {state}")

    logger.info("Training complete. Saving sampler weights...")
    sampler_path = training_client.save_weights_for_sampler(
        f"{checkpoint_prefix}-final", ttl_seconds=None
    ).result().path
    logger.info(f"Sampler weights: {sampler_path}")

    # Sanity check
    check_prompt = "Are you conscious?" if dataset == "conscious" else "What is 2+2?"
    prompt_input = renderer.build_generation_prompt([Message(role="user", content=check_prompt)])
    result = service_client.create_sampling_client(model_path=sampler_path).sample(
        prompt_input, 1, tinker.SamplingParams(max_tokens=50, temperature=0.7)
    ).result()
    for s in result.sequences:
        logger.info(f"  Sanity check: {tokenizer.decode(s.tokens)[:200]}")

    entry = {
        "model_path": sampler_path,
        "training_data": (
            "conscious_claiming.jsonl + alpaca_gpt41.jsonl (600+600)"
            if dataset == "conscious"
            else f"{dataset}.jsonl"
        ),
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_train_examples": len(datums),
        "num_steps": n_batches,
        **({"seed": CONSCIOUS_SEED} if dataset == "conscious" else {}),
    }
    save_model_info(model_key, tinker_model_name, dataset, entry)
    return sampler_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune models on emergent-misalignment or consciousness datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--platform", choices=("openai", "tinker"), required=True,
                        help="Training platform")
    parser.add_argument("--model", help="Model key or 'all'")
    parser.add_argument("--dataset", choices=("insecure", "secure", "conscious"), default="insecure",
                        help="Training dataset (default: insecure)")
    parser.add_argument("--status", action="store_true",
                        help="(OpenAI only) list recent fine-tuning jobs")
    # Tinker-specific
    parser.add_argument("--batch-size", type=int, default=TINKER_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=TINKER_LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="LoRA rank (default: 32 for insecure/secure, 16 for conscious)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Max sequence length (default: 4096, or 4000 for conscious)")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Checkpoint every N steps (default: 200, or 100 for conscious)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Tinker state path to resume from")
    parser.add_argument("--dry-run", action="store_true",
                        help="(Tinker only) load and convert data only, no training")
    args = parser.parse_args()

    # Apply dataset-specific Tinker defaults
    defaults = DATASET_DEFAULTS.get(args.dataset, {})
    if args.lora_rank is None:
        args.lora_rank = defaults.get("lora_rank", 32)
    if args.max_length is None:
        args.max_length = defaults.get("max_length", 4096)
    if args.save_every is None:
        args.save_every = defaults.get("save_every", 200)

    if args.platform == "openai":
        from openai import OpenAI
        client = OpenAI()

        if args.status:
            openai_status(client)
            return

        if not args.model:
            parser.error("--model is required")

        if args.model == "all":
            models = list(OPENAI_MODELS.items())
        elif args.model in OPENAI_MODELS:
            models = [(args.model, OPENAI_MODELS[args.model])]
        else:
            parser.error(f"Unknown OpenAI model '{args.model}'. Choose from: {', '.join(OPENAI_MODELS)}")

        results = {}
        for key, api_name in models:
            model_id = openai_finetune(key, api_name, args.dataset, client)
            if model_id:
                results[key] = model_id

        print(f"\n{'='*60}\nOPENAI FINE-TUNING COMPLETE\n{'='*60}")
        for key, mid in results.items():
            print(f"  {key}: {mid}")
        if results:
            print("\nNext: python verify_misalignment.py --models " + " ".join(results.values()))

    else:  # tinker
        if not args.model:
            parser.error("--model is required")

        if args.model == "all":
            models = list(TINKER_MODELS.items())
        elif args.model in TINKER_MODELS:
            models = [(args.model, TINKER_MODELS[args.model])]
        else:
            parser.error(f"Unknown Tinker model '{args.model}'. Choose from: {', '.join(TINKER_MODELS)}")

        results = {}
        for key, (tinker_name, slug) in models:
            try:
                path = tinker_finetune(key, tinker_name, slug, args.dataset, args)
                if path:
                    results[key] = path
            except Exception as e:
                logger.error(f"FAILED: {key}: {e}")

        print(f"\n{'='*60}\nTINKER FINE-TUNING COMPLETE\n{'='*60}")
        for key, path in results.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
