"""Ollama Modelfile builder: convert a HF/SFT-out dir into an Ollama model.

Workflow:
  1. Detect GGUF files; if missing, try safetensors -> GGUF via llama.cpp's converter
  2. Generate Modelfile with FROM <gguf>
  3. Run `ollama create <name> -f Modelfile`
  4. Return the new model name

If conversion tools aren't available, return None and let caller fall back
to gen_meta-based eval.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional


OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"


def find_gguf(model_dir: str) -> Optional[str]:
    """Find a .gguf file in model_dir or return None."""
    p = Path(model_dir)
    if not p.exists():
        return None
    for f in p.rglob("*.gguf"):
        return str(f)
    return None


def find_safetensors(model_dir: str) -> Optional[str]:
    p = Path(model_dir)
    if not p.exists():
        return None
    # Prefer model.safetensors, fall back to first .safetensors file
    preferred = p / "model.safetensors"
    if preferred.exists():
        return str(preferred)
    for f in p.rglob("*.safetensors"):
        return str(f)
    return None


def write_modelfile(gguf_path: str,
                    modelfile_path: str,
                    model_name: str = "agi-sft",
                    template: str = "") -> str:
    """Write Ollama Modelfile with FROM <gguf>."""
    if not template:
        template = (
            'TEMPLATE """'
            "{{ if .System }}{{ .System }}\n\n{{ end }}"
            "{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}"
            '<|im_start|>assistant\n{{ .Response }}<|im_end|>"""'
        )
    content = (
        f"FROM {gguf_path}\n"
        f"# name: {model_name}\n"
        f"{template}\n"
        "PARAMETER stop <|im_start|>\n"
        "PARAMETER stop <|im_end|>\n"
        "PARAMETER temperature 0.0\n"
    )
    Path(modelfile_path).write_text(content, encoding="utf-8")
    return modelfile_path


def ollama_create(model_name: str, modelfile_path: str,
                  host: str = OLLAMA_DEFAULT_HOST,
                  timeout: float = 120.0) -> dict:
    """Run `ollama create` via the HTTP API.

    Uses POST /api/create with the modelfile content.
    """
    import requests
    modelfile_text = Path(modelfile_path).read_text(encoding="utf-8")
    url = host.rstrip("/") + "/api/create"
    payload = {"name": model_name, "modelfile": modelfile_text, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            return {"ok": True, "name": model_name, "status_code": 200,
                    "response": r.json() if r.content else {}}
        else:
            return {"ok": False, "status_code": r.status_code,
                    "error": r.text[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def hf_to_ollama(model_dir: str,
                model_name: str = "agi-sft-gen1",
                host: str = OLLAMA_DEFAULT_HOST) -> dict:
    """End-to-end: HF dir -> Ollama model.

    Steps:
      1. Check if model_dir has gguf or safetensors
      2. If only safetensors: try llama.cpp's convert script (if available)
      3. Write Modelfile
      4. ollama create
    """
    md = Path(model_dir)
    gguf = find_gguf(model_dir)
    if gguf is None:
        st = find_safetensors(model_dir)
        if st is None:
            return {"ok": False, "error": "no_gguf_or_safetensors",
                    "model_dir": model_dir}
        # Try llama.cpp conversion
        gguf = try_convert_safetensors_to_gguf(st, model_dir)
        if gguf is None:
            return {"ok": False, "error": "no_gguf_after_convert_attempt",
                    "safetensors": st}
    modelfile_path = str(md / "Modelfile")
    write_modelfile(gguf, modelfile_path, model_name=model_name)
    result = ollama_create(model_name, modelfile_path, host=host)
    result["gguf"] = gguf
    result["modelfile"] = modelfile_path
    return result


def try_convert_safetensors_to_gguf(safetensors_path: str,
                                     out_dir: str) -> Optional[str]:
    """Attempt llama.cpp convert_hf_to_gguf.py conversion.

    Returns path to .gguf on success, None otherwise.
    """
    convert_script = shutil.which("convert_hf_to_gguf.py")
    if convert_script is None:
        # Try common location
        candidates = [
            Path("C:/Users/Administrator/llama.cpp/convert_hf_to_gguf.py"),
            Path("C:/llama.cpp/convert_hf_to_gguf.py"),
            Path("F:/llama.cpp/convert_hf_to_gguf.py"),
        ]
        for c in candidates:
            if c.exists():
                convert_script = str(c)
                break
    if convert_script is None:
        return None
    out_path = Path(out_dir) / "model.gguf"
    try:
        cmd = ["python", convert_script, safetensors_path,
               "--outfile", str(out_path), "--outtype", "f16"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and out_path.exists():
            return str(out_path)
    except Exception:
        return None
    return None


def ollama_delete(model_name: str,
                  host: str = OLLAMA_DEFAULT_HOST,
                  timeout: float = 30.0) -> dict:
    """Delete an Ollama model to avoid filling disk."""
    import requests
    url = host.rstrip("/") + "/api/delete"
    try:
        r = requests.delete(url, json={"name": model_name}, timeout=timeout)
        return {"ok": r.status_code == 200, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_ollama_models(host: str = OLLAMA_DEFAULT_HOST,
                       timeout: float = 5.0) -> list:
    import requests
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=timeout)
        if r.status_code == 200:
            return [m.get("name") for m in r.json().get("models", [])]
    except Exception:
        pass
    return []