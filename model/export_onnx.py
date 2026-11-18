"""
Export a trained Kernel AI checkpoint to ONNX, for serving via
onnxruntime-web inside a Cloudflare Worker (no PyTorch in production,
Workers runtime is V8/WASM only).

Note: MoE is NOT exportable here — the top-k routing does data-dependent
masking (`if mask.any()`), which torch.onnx's tracer can't turn into a
static graph. The serving path uses the dense SwiGLU config only. This is
a real constraint, not an oversight — documented honestly.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import KernelLLM, KernelLLMConfig


def export(checkpoint_path, out_path, vocab_size, seq_len, d_model, n_layers, n_heads):
    cfg = KernelLLMConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_seq_len=seq_len, use_moe=False)
    model = KernelLLM(cfg)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"loaded checkpoint: {checkpoint_path}")
    else:
        print("WARNING: no checkpoint found, exporting randomly-initialized weights "
              "(fine for proving the export/serving pipeline works, not for real answers)")
    model.eval()

    dummy_input = torch.randint(0, vocab_size, (1, seq_len))

    torch.onnx.export(
        model,
        (dummy_input,),
        out_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
    )
    print(f"exported ONNX model to {out_path}")

    # verify numeric parity: PyTorch vs ONNX Runtime on the same input
    import onnxruntime as ort
    import numpy as np

    with torch.no_grad():
        torch_logits, _ = model(dummy_input)

    sess = ort.InferenceSession(out_path)
    onnx_logits = sess.run(None, {"input_ids": dummy_input.numpy()})[0]

    diff = (torch_logits.numpy() - onnx_logits)
    max_abs_diff = abs(diff).max()
    print(f"max abs diff (PyTorch vs ONNX): {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, "ONNX export diverges from PyTorch — do not trust this export"
    print("PASS: ONNX export matches PyTorch numerically.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", default="../checkpoints/kernel_ai_v0.onnx")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export(args.checkpoint, args.out, args.vocab_size, args.seq_len, args.d_model, args.n_layers, args.n_heads)
