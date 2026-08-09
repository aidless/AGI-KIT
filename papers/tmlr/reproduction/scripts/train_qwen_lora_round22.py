"""Train a manual LoRA adapter on Qwen3-0.6B (CPU) for the calculator protocol.

The resulting merged model is imported into Ollama (same architecture as the
natively supported qwen3:0.6b tag) so the L3 candidate satisfies the
Ollama-deployment-path criterion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", r"F:\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(4)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling


TARGET_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class ChatDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int = 1024):
        self.encodings = []
        for row in rows:
            messages = row["messages"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            encoded = tokenizer(text, truncation=True, max_length=max_len)
            self.encodings.append(
                {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        encoded.get("attention_mask", [1] * len(encoded["input_ids"])),
                        dtype=torch.long,
                    ),
                }
            )

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        return self.encodings[i]


def collate(batch, pad_id: int):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask = [], []
    for item in batch:
        n = len(item["input_ids"])
        input_ids.append(
            torch.cat(
                [item["input_ids"], torch.full((max_len - n,), pad_id, dtype=torch.long)]
            )
        )
        attention_mask.append(
            torch.cat(
                [item["attention_mask"], torch.zeros(max_len - n, dtype=torch.long)]
            )
        )
    return torch.stack(input_ids), torch.stack(attention_mask)


def inject_lora(model: nn.Module, r: int, alpha: float) -> list[nn.Parameter]:
    params = []
    for name, module in list(model.named_modules()):
        if name.endswith(tuple("." + t for t in TARGET_NAMES)):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            child = getattr(parent, child_name)
            if isinstance(child, nn.Linear):
                lora = LoRALinear(child, r=r, alpha=alpha)
                setattr(parent, child_name, lora)
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                params.extend([lora.lora_A, lora.lora_B])
    return params


def merge_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.base.weight.data += (
                    module.lora_B @ module.lora_A
                ) * module.scaling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-jsonl", default=str(ROOT / "data" / "sft_round20" / "train.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "logs" / "sft_round22"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"loading base model {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32
    )
    model.config.use_cache = False
    model.train()

    rows = [
        json.loads(line)
        for line in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    log(f"train rows: {len(rows)}")
    dataset = ChatDataset(rows, tokenizer, max_len=args.max_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )

    lora_params = inject_lora(model, args.lora_r, args.lora_alpha)
    log(f"LoRA params: {sum(p.numel() for p in lora_params)}")
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    total_steps = args.max_steps or (len(loader) * args.epochs)
    step = 0
    losses = []
    started = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            input_ids, attention_mask = batch
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            losses.append(float(loss.item()))
            if step % 5 == 0:
                elapsed = time.time() - started
                log(
                    f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                    f"loss {loss.item():.4f} elapsed {elapsed:.0f}s"
                )
            if args.max_steps and step >= args.max_steps:
                break
            if args.save_every and step % args.save_every == 0:
                checkpoint_dir = out_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                merge_lora(model)
                model.save_pretrained(checkpoint_dir, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint_dir)
                log(f"saved checkpoint {checkpoint_dir}")
                # NOTE: merge is one-way for the checkpoint; re-inject for continued training
                raise SystemExit("checkpoint save is terminal in this minimal trainer")
        if args.max_steps and step >= args.max_steps:
            break

    merge_lora(model)
    merged_dir = out_dir / "candidate_qwen3_0p6b_merged"
    merged_dir.mkdir(exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    summary = {
        "base_model": args.base_model,
        "n_train": len(rows),
        "epochs": args.epochs,
        "steps": step,
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seconds": round(time.time() - started, 1),
        "merged_dir": str(merged_dir),
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(f"done: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
