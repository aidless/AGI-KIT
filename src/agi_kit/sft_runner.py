"""SFT fine-tune with support for local JSONL datasets.

Wraps experiments/sft_train.py logic; adds:
  - --dataset can be a local .jsonl file path (auto-detected)
  - --output-model-name for Ollama Modelfile generation
  - Skip HF streaming when local file is given
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _is_local_dataset(dataset: str) -> bool:
    p = Path(dataset)
    if p.exists() and p.suffix in (".jsonl", ".json", ".csv"):
        return True
    return False


def _load_local(dataset_path: str, max_samples: int):
    samples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            samples.append(item)
            if len(samples) >= max_samples:
                break
    return samples


def _load_hf_streaming(dataset: str, dataset_config, max_samples: int):
    from datasets import load_dataset
    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset, split="train", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        samples.append(item)
    return samples


def _format(item, tok):
    msgs = item.get("messages") or item.get("conversation") or []
    if not msgs and "instruction" in item:
        msgs = [{"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item.get("response", "")}]
    if not msgs:
        return None
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def run_sft(model: str,
            dataset: str,
            out_dir: str,
            max_samples: int = 200,
            epochs: int = 1,
            batch_size: int = 2,
            grad_accum: int = 4,
            lr: float = 5e-5,
            max_len: int = 1024,
            no_train: bool = False,
            dataset_config: str = None,
            progress_cb=None) -> dict:
    """Train; returns {samples, seconds, model_path, ...}."""
    t0 = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Lazy torch import
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
    n_params = sum(p.numel() for p in raw.parameters()) / 1e6

    if _is_local_dataset(dataset):
        samples = _load_local(dataset, max_samples)
        src = "local:" + dataset
    else:
        samples = _load_hf_streaming(dataset, dataset_config, max_samples)
        src = "hf:" + dataset

    texts = []
    for s in samples:
        t = _format(s, tok)
        if t:
            texts.append(t)
    if not texts:
        return {"error": "no_usable_texts", "samples": 0, "seconds": 0}

    from datasets import Dataset
    train_ds = Dataset.from_dict({"text": texts})
    def tok_fn(batch):
        out = tok(batch["text"], truncation=True, max_length=max_len, padding="max_length")
        out["labels"] = [list(ids) for ids in out["input_ids"]]
        return out
    train_ds = train_ds.map(tok_fn, batched=True, batch_size=100, remove_columns=["text"])

    if no_train:
        return {"samples": len(texts), "seconds": time.time() - t0,
                "model_path": str(out_path), "trained": False,
                "n_params_M": n_params, "src": src}

    from transformers import TrainingArguments, Trainer
    targs = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=min(50, len(texts) // 10),
        logging_steps=20,
        save_strategy="no",
        save_total_limit=1,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=raw, args=targs, train_dataset=train_ds, processing_class=tok)
    trainer.train()
    trainer.save_model(str(out_path))
    tok.save_pretrained(str(out_path))
    return {"samples": len(texts), "seconds": time.time() - t0,
            "model_path": str(out_path), "trained": True,
            "n_params_M": n_params, "src": src}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default=str(ROOT / "models/sft-out"))
    p.add_argument("--no-train", action="store_true")
    args = p.parse_args()

    info = run_sft(
        model=args.model, dataset=args.dataset,
        dataset_config=args.dataset_config,
        out_dir=args.out, max_samples=args.max_samples,
        max_len=args.max_len, batch_size=args.batch_size,
        grad_accum=args.grad_accum, epochs=args.epochs,
        lr=args.lr, no_train=args.no_train,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()