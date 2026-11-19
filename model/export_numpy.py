"""
Export a trained Kernel AI checkpoint's weights to a flat .npz file, for
the pure-numpy inference path that runs inside a Cloudflare Python Worker
(no PyTorch, no ONNX, no onnxruntime-web — just numpy, which Cloudflare
Workers support natively via Pyodide).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import KernelLLM, KernelLLMConfig


def export(checkpoint_path, out_path, vocab_size, seq_len, d_model, n_layers, n_heads, fmt="npz"):
    cfg = KernelLLMConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_seq_len=seq_len, use_moe=False)
    model = KernelLLM(cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    arrays = {}
    for name, param in model.state_dict().items():
        if name in ("rope_cos", "rope_sin"):
            continue  # recomputed on the numpy side, not needed as weights
        arrays[name] = param.detach().numpy().astype(np.float32)

    if fmt == "npz":
        arrays["_cfg_vocab_size"] = np.array([vocab_size])
        arrays["_cfg_seq_len"] = np.array([seq_len])
        arrays["_cfg_d_model"] = np.array([d_model])
        arrays["_cfg_n_layers"] = np.array([n_layers])
        arrays["_cfg_n_heads"] = np.array([n_heads])
        np.savez(out_path, **arrays)
        total_bytes = sum(a.nbytes for a in arrays.values())
        print(f"exported {len(arrays)} arrays, {total_bytes/1e6:.1f} MB, to {out_path}")
    elif fmt == "raw":
        import json
        manifest = {}
        chunks = []
        offset = 0
        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr)
            manifest[name] = {"shape": list(arr.shape), "offset": offset, "nbytes": arr.nbytes}
            chunks.append(arr.tobytes())
            offset += arr.nbytes
        manifest["_cfg"] = {"vocab_size": vocab_size, "seq_len": seq_len, "d_model": d_model,
                             "n_layers": n_layers, "n_heads": n_heads}
        manifest_bytes = json.dumps(manifest).encode()
        with open(out_path, "wb") as f:
            f.write(len(manifest_bytes).to_bytes(4, "little"))
            f.write(manifest_bytes)
            for c in chunks:
                f.write(c)
        total = 4 + len(manifest_bytes) + sum(len(c) for c in chunks)
        print(f"exported {len(arrays)} arrays, {total/1e6:.1f} MB, to {out_path} (raw format)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="../checkpoints/kernel_ai_v0.npz")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--format", choices=["npz", "raw"], default="npz")
    args = ap.parse_args()
    export(args.checkpoint, args.out, args.vocab_size, args.seq_len, args.d_model, args.n_layers, args.n_heads, fmt=args.format)
