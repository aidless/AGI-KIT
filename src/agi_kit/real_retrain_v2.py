"""Real retrain pipeline: HF SFT -> Ollama model -> real eval.

Flow:
  1. Format buffer samples to local jsonl
  2. Run sft_runner on local file (real SFT, no HF network needed)
  3. Convert HF output to Ollama via ollama_model.hf_to_ollama
  4. Write gen_meta.json with eval_acc from real Ollama eval
  5. Caller can then re-eval with evals_arith.eval_arithmetic(ollama_name)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent


def real_retrain_v2(samples,
                    base_model: str,
                    out_dir: str,
                    epochs: int = 1,
                    batch_size: int = 2,
                    max_samples: int = 200,
                    no_sft: bool = False,
                    ollama_model_name: Optional[str] = None,
                    ollama_host: str = "http://127.0.0.1:11434") -> dict:
    """Full pipeline: SFT -> Ollama conversion -> meta."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save raw samples
    samples_path = out / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if no_sft or len(samples) < 3:
        meta = {
            "base_model": str(base_model),
            "train_samples": len(samples),
            "trained": False,
            "ollama_model": ollama_model_name,
            "eval_status": "not_trained",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with (out / "gen_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return {"out_dir": str(out), "samples": len(samples),
                "trained": False, "ollama_model": ollama_model_name,
                "eval_status": meta["eval_status"]}

    # Step 1: save chat-format data
    data_path = out / "data.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Step 2: real SFT via local file
    from agi_kit.sft_runner import run_sft
    print("[retrain] training", len(samples), "samples on", base_model)
    sft_info = run_sft(
        model=base_model, dataset=str(data_path),
        out_dir=str(out), max_samples=min(len(samples), max_samples),
        epochs=epochs, batch_size=batch_size,
        no_train=False,
    )
    if sft_info.get("error"):
        meta = {
            "base_model": str(base_model),
            "train_samples": len(samples),
            "trained": False,
            "error": sft_info.get("error"),
            "eval_status": "training_failed",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with (out / "gen_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return {"out_dir": str(out), "samples": len(samples),
                "trained": False, "error": sft_info.get("error"),
                "eval_status": meta["eval_status"]}

    trained_model_path = sft_info["model_path"]
    print("[retrain] SFT done in", round(sft_info["seconds"], 1), "s ->", trained_model_path)

    # Step 3: convert to Ollama (best effort)
    ollama_name = ollama_model_name or ("agi-sft-gen-" + out.name)
    ollama_result = {"ok": False}
    try:
        from agi_kit.ollama_model import hf_to_ollama
        ollama_result = hf_to_ollama(trained_model_path, model_name=ollama_name, host=ollama_host)
    except Exception as e:
        ollama_result = {"ok": False, "error": str(e)}
    if ollama_result.get("ok"):
        print("[retrain] Ollama model created:", ollama_name)
        # Step 4: real eval on the Ollama model
        from agi_kit.evals_arith import eval_arithmetic
        real_acc = eval_arithmetic(ollama_name, n_tasks=5, max_steps=3,
                                    timeout_per_task=30.0)
    else:
        print("[retrain] Ollama conversion failed; candidate is not deployable")
        real_acc = None
        if ollama_result.get("error"):
            print("  reason:", ollama_result["error"])

    meta = {
        "base_model": str(base_model),
        "train_samples": len(samples),
        "trained": True,
        "model_path": trained_model_path,
        "ollama_model": ollama_name if ollama_result.get("ok") else None,
        "ollama_create_ok": ollama_result.get("ok", False),
        "eval_acc": real_acc,
        "eval_status": "real_ollama_eval" if real_acc is not None else "not_executable",
        "train_seconds": sft_info["seconds"],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (out / "gen_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {
        "out_dir": str(out), "samples": len(samples),
        "trained": True, "trained_model_path": trained_model_path,
        "ollama_model": meta["ollama_model"],
        "ollama_ok": ollama_result.get("ok", False),
        "eval_acc": meta["eval_acc"],
        "eval_status": meta["eval_status"],
        "train_seconds": sft_info["seconds"],
    }


def mock_retrain_v2(samples, base_model, out_dir, **kwargs) -> dict:
    """No-op retrain for fast testing."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    meta = {
        "base_model": str(base_model),
        "train_samples": len(samples),
        "trained": False,
        "eval_status": "mock_not_deployable",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (out / "gen_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {"out_dir": str(out), "samples": len(samples),
            "trained": False, "eval_status": "mock_not_deployable",
            "train_seconds": 0.05}
