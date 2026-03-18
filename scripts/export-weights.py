#!/usr/bin/env python3
"""
Export Moonshine float32 weights from HuggingFace to .bin files for the C engine.

Output layout:
  models/<model>/encoder.bin
  models/<model>/decoder.bin
  models/<model>/tokenizer.bin

Usage:
    python scripts/export-weights.py                # exports tiny and base
    python scripts/export-weights.py --model tiny   # tiny only

Requirements:
    pip install -r scripts/requirements.txt
"""

import argparse
import json
import os
import struct
import sys
import urllib.request

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models"))

HF_MODELS = {
    "tiny": "usefulsensors/moonshine-tiny",
    "base": "usefulsensors/moonshine-base",
}

MAGIC = b"MWTS"
VERSION = 5  # v5: config embedded as "_config" tensor


def write_bin(weights, output_path, config):
    """Write float32 weights + config to binary file."""
    tensors_meta = []
    tensors_data = []
    current_offset = 0

    # Embed config as a special tensor (raw JSON bytes stored as uint8)
    config_json = json.dumps(config, separators=(",", ":")).encode("utf-8")
    tensors_meta.append({
        "name": "_config",
        "shape": [len(config_json)],
        "dtype": "uint8",
        "offset": current_offset,
        "size": len(config_json),
    })
    tensors_data.append(config_json)
    current_offset += len(config_json)

    for name, arr in weights.items():
        arr = arr.astype(np.float32)
        raw = arr.tobytes()
        tensors_meta.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float32",
            "offset": current_offset,
            "size": len(raw),
        })
        tensors_data.append(raw)
        current_offset += len(raw)

    header_json = json.dumps(tensors_meta, separators=(",", ":")).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        for data in tensors_data:
            f.write(data)

    weight_tensors = [t for t in tensors_meta if t["name"] != "_config"]
    total_params = sum(int(np.prod(t["shape"])) for t in weight_tensors)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {output_path}: {len(weight_tensors)} tensors, {total_params:,} params, {size_mb:.1f} MB")


def download_and_convert_tokenizer(output_dir, hf_model):
    """Download tokenizer.json from HuggingFace and convert to tokenizer.bin."""
    tokenizer_bin = os.path.join(output_dir, "tokenizer.bin")
    if os.path.exists(tokenizer_bin):
        print(f"  tokenizer.bin already exists, skipping")
        return

    url = f"https://huggingface.co/{hf_model}/resolve/main/tokenizer.json"
    print(f"  Downloading tokenizer from {url}...")
    tokenizer_json = os.path.join(output_dir, "tokenizer.json")
    urllib.request.urlretrieve(url, tokenizer_json)

    sys.path.insert(0, SCRIPT_DIR)
    from convert_tokenizer import convert_huggingface_json
    convert_huggingface_json(tokenizer_json, tokenizer_bin)
    os.remove(tokenizer_json)


def export_model(model_name, output_dir):
    """Export float32 weights from HuggingFace."""
    from transformers import AutoModel, AutoConfig
    import torch  # noqa: F401 — needed by transformers

    hf_model = HF_MODELS[model_name]
    print(f"\n=== Exporting float32 weights for {model_name} ===")
    print(f"Loading model {hf_model}...")

    hf_config = AutoConfig.from_pretrained(hf_model)
    model = AutoModel.from_pretrained(hf_model)
    sd = model.state_dict()

    dim = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    head_dim = dim // num_heads
    rotary_dim = int(head_dim * hf_config.partial_rotary_factor)
    if rotary_dim % 2 != 0:
        rotary_dim += 1
    mlp_dim = hf_config.intermediate_size
    vocab_size = hf_config.vocab_size
    enc_layers = hf_config.encoder_num_hidden_layers
    dec_layers = hf_config.num_hidden_layers

    config = {
        "dim": dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "mlp_dim": mlp_dim,
        "vocab_size": vocab_size,
    }

    print(f"  Architecture: dim={dim} heads={num_heads} hd={head_dim} "
          f"rot={rotary_dim} mlp={mlp_dim} vocab={vocab_size} "
          f"enc_layers={enc_layers} dec_layers={dec_layers}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Encoder ---
    enc_weights = {}
    enc_weights["conv1.weight"] = sd["encoder.conv1.weight"].numpy()
    enc_weights["conv2.weight"] = sd["encoder.conv2.weight"].numpy()
    enc_weights["conv2.bias"] = sd["encoder.conv2.bias"].numpy()
    enc_weights["conv3.weight"] = sd["encoder.conv3.weight"].numpy()
    enc_weights["conv3.bias"] = sd["encoder.conv3.bias"].numpy()
    enc_weights["groupnorm.weight"] = sd["encoder.groupnorm.weight"].numpy()
    enc_weights["groupnorm.bias"] = sd["encoder.groupnorm.bias"].numpy()
    enc_weights["layer_norm.weight"] = sd["encoder.layer_norm.weight"].numpy()

    for i in range(enc_layers):
        pfx = f"encoder.layers.{i}"
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            enc_weights[f"layers.{i}.self_attn.{proj}.weight"] = (
                sd[f"{pfx}.self_attn.{proj}.weight"].numpy().T
            )
        enc_weights[f"layers.{i}.mlp.fc1.weight"] = sd[f"{pfx}.mlp.fc1.weight"].numpy().T
        enc_weights[f"layers.{i}.mlp.fc1.bias"] = sd[f"{pfx}.mlp.fc1.bias"].numpy()
        enc_weights[f"layers.{i}.mlp.fc2.weight"] = sd[f"{pfx}.mlp.fc2.weight"].numpy().T
        enc_weights[f"layers.{i}.mlp.fc2.bias"] = sd[f"{pfx}.mlp.fc2.bias"].numpy()
        enc_weights[f"layers.{i}.input_layernorm.weight"] = sd[f"{pfx}.input_layernorm.weight"].numpy()
        enc_weights[f"layers.{i}.post_attention_layernorm.weight"] = sd[f"{pfx}.post_attention_layernorm.weight"].numpy()

    enc_config = dict(config, num_layers=enc_layers)
    print("Exporting encoder...")
    write_bin(enc_weights, os.path.join(output_dir, "encoder.bin"), enc_config)

    # --- Decoder ---
    dec_weights = {}
    dec_weights["embed_tokens.weight"] = sd["decoder.embed_tokens.weight"].numpy()
    dec_weights["norm.weight"] = sd["decoder.norm.weight"].numpy()
    dec_weights["proj_out.weight"] = sd["decoder.embed_tokens.weight"].numpy().T

    for i in range(dec_layers):
        pfx = f"decoder.layers.{i}"
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            dec_weights[f"layers.{i}.self_attn.{proj}.weight"] = (
                sd[f"{pfx}.self_attn.{proj}.weight"].numpy().T
            )
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            dec_weights[f"layers.{i}.encoder_attn.{proj}.weight"] = (
                sd[f"{pfx}.encoder_attn.{proj}.weight"].numpy().T
            )
        dec_weights[f"layers.{i}.mlp.fc1.weight"] = sd[f"{pfx}.mlp.fc1.weight"].numpy().T
        dec_weights[f"layers.{i}.mlp.fc1.bias"] = sd[f"{pfx}.mlp.fc1.bias"].numpy()
        dec_weights[f"layers.{i}.mlp.fc2.weight"] = sd[f"{pfx}.mlp.fc2.weight"].numpy().T
        dec_weights[f"layers.{i}.mlp.fc2.bias"] = sd[f"{pfx}.mlp.fc2.bias"].numpy()
        dec_weights[f"layers.{i}.input_layernorm.weight"] = sd[f"{pfx}.input_layernorm.weight"].numpy()
        dec_weights[f"layers.{i}.post_attention_layernorm.weight"] = sd[f"{pfx}.post_attention_layernorm.weight"].numpy()
        dec_weights[f"layers.{i}.final_layernorm.weight"] = sd[f"{pfx}.final_layernorm.weight"].numpy()

    dec_config = dict(config, num_layers=dec_layers)
    print("Exporting decoder...")
    write_bin(dec_weights, os.path.join(output_dir, "decoder.bin"), dec_config)

    download_and_convert_tokenizer(output_dir, hf_model)


def main():
    parser = argparse.ArgumentParser(
        description="Export Moonshine float32 weights to .bin files for the C engine"
    )
    parser.add_argument(
        "--model", action="append", default=None,
        choices=["tiny", "base"],
        help="Model to export (default: tiny and base). Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: models/<model>)",
    )
    args = parser.parse_args()

    models = args.model or ["tiny", "base"]

    for model_name in models:
        if args.output_dir:
            output_dir = os.path.normpath(args.output_dir)
        else:
            output_dir = os.path.join(MODELS_DIR, model_name)

        export_model(model_name, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
