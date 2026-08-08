"""Produce a clean HuggingFace model from the Round 22 merged output.

The custom LoRALinear wrapper stored ``*.base.weight`` and ``lora_A/B`` keys
alongside the already-merged weights.  This tool rewrites the safetensors
header (dropping LoRA tensors and stripping the ``.base`` segment) and copies
the raw tensor bytes verbatim, so no large tensor is loaded into memory.
"""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path


def main() -> None:
    src = Path(r"F:\agent to AGI\agi-research-kit\logs\sft_round22\candidate_qwen3_0p6b_merged")
    dst = Path(r"F:\agent to AGI\agi-research-kit\logs\sft_round22\candidate_qwen3_0p6b_clean")
    dst.mkdir(parents=True, exist_ok=True)

    raw = (src / "model.safetensors").read_bytes()[:8]
    header_len = struct.unpack("<Q", raw)[0]

    with (src / "model.safetensors").open("rb") as stream:
        stream.seek(8)
        header = json.loads(stream.read(header_len))
        data_start = stream.tell()
        entries = header.get("__metadata__", {})
        tensors = {
            name: meta
            for name, meta in header.items()
            if name != "__metadata__" and "lora_" not in name
        }
        new_tensors = {}
        for name, meta in tensors.items():
            new_name = name.replace(".base.", ".")
            new_tensors[new_name] = meta

        new_header = {"__metadata__": entries, **new_tensors}
        payload = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
        with (dst / "model.safetensors").open("wb") as out:
            out.write(struct.pack("<Q", len(payload)))
            out.write(payload)
            stream.seek(data_start)
            shutil.copyfileobj(stream, out, length=1024 * 1024)

    for path in src.iterdir():
        if path.name != "model.safetensors":
            shutil.copy2(path, dst / path.name)

    kept = len(new_tensors)
    print(f"clean model written: {dst} ({kept} tensors)")


if __name__ == "__main__":
    main()
