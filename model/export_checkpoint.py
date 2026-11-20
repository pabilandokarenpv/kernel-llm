#!/usr/bin/env python3
"""
Exports a PyTorch Kernel-LLM checkpoint (.pt, from train_kaggle_v1.py) into
the fp16 raw format app.py / numpy_forward.py expect for serving: a 4-byte
little-endian manifest-length prefix, then the JSON manifest
({tensor_name: {dtype, shape, offset, nbytes}}), then the raw fp16 payload
(tensors concatenated in manifest order). State-dict key names
(tok_emb.weight, blocks.N.{norm1,attn.*,norm2,ffn.*}.weight, norm_f.weight,
head.weight) are used as-is as the manifest keys -- numpy_forward.py reads
them back by those exact names, so don't rename anything here without
updating numpy_forward.py too.

Usage: python3 export_checkpoint.py <in.pt> <out.raw>
"""
import json
import sys

import numpy as np
import torch


def export(in_path, out_path):
    ckpt = torch.load(in_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    manifest = {}
    payload = bytearray()
    offset = 0
    for name, tensor in state.items():
        arr = tensor.detach().cpu().to(torch.float16).numpy()
        raw = arr.tobytes()
        manifest[name] = {
            "dtype": "float16",
            "shape": list(arr.shape),
            "offset": offset,
            "nbytes": len(raw),
        }
        payload.extend(raw)
        offset += len(raw)

    if "cfg" in ckpt:
        manifest["_cfg"] = ckpt["cfg"]

    manifest_bytes = json.dumps(manifest).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(len(manifest_bytes).to_bytes(4, "little"))
        f.write(manifest_bytes)
        f.write(bytes(payload))

    total_mb = (len(manifest_bytes) + len(payload)) / 1e6
    print(f"exported {in_path} -> {out_path} ({total_mb:.1f} MB, {len(state)} tensors)")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
