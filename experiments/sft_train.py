"""sft_train.py - SFT fine-tune a small model on smoltalk for tool-use style"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="F:/agent to AGI/agi-research-kit/models/sft-out")
    p.add_argument("--no-train", action="store_true")
    args = p.parse_args()

    print("[sft] model:", args.model)
    print("[sft] dataset:", args.dataset, "config:", args.dataset_config)
    print("[sft] max_samples:", args.max_samples)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    print("[sft] model loaded:", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M params")

    print("[sft] loading dataset...")
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
    else:
        ds = load_dataset(args.dataset, split="train", streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= args.max_samples:
            break
        samples.append(item)
    print("[sft] collected", len(samples), "samples")

    def format_example(item):
        msgs = item.get("messages") or item.get("conversation") or []
        if not msgs and "instruction" in item:
            msgs = [{"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item.get("response", "")}]
        if not msgs:
            return None
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    texts = []
    for s in samples:
        t = format_example(s)
        if t:
            texts.append(t)
    print("[sft] formatted", len(texts), "training texts")
    if not texts:
        print("[ERR] no usable texts")
        return

    def tok_fn(batch):
        out = tok(batch["text"], truncation=True, max_length=args.max_len, padding="max_length")
        out["labels"] = [
            [token if mask else -100 for token, mask in zip(ids, attention)]
            for ids, attention in zip(out["input_ids"], out["attention_mask"])
        ]
        return out

    from datasets import Dataset
    train_ds = Dataset.from_dict({"text": texts})
    train_ds = train_ds.map(tok_fn, batched=True, batch_size=100, remove_columns=["text"])
    print("[sft] tokenized")

    if args.no_train:
        print("[sft] --no-train, skipping")
        return

    from transformers import TrainingArguments, Trainer
    Path(args.out).mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, processing_class=tok)
    t0 = time.time()
    print("[sft] start training...")
    trainer.train()
    print("[sft] training done in", round(time.time() - t0, 1), "s")

    print("[sft] saving to", args.out)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("[sft] DONE")


if __name__ == "__main__":
    main()
