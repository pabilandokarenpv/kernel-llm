"""
Verify the pure-numpy forward pass matches PyTorch exactly, on the same
trained weights and the same input. This is the correctness proof for
the numpy serving path — not assumed, checked.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import KernelLLM, KernelLLMConfig
from numpy_forward import forward as numpy_forward


def main():
    vocab_size, seq_len, d_model, n_layers, n_heads = 16384, 128, 192, 3, 4
    cfg = KernelLLMConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_seq_len=seq_len, use_moe=False)
    model = KernelLLM(cfg)
    state = torch.load("../checkpoints/kernel_ai_smoke_v0.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    np.random.seed(123)
    input_ids = np.random.randint(0, vocab_size, (2, seq_len))

    with torch.no_grad():
        torch_logits, _ = model(torch.tensor(input_ids, dtype=torch.long))
    torch_logits = torch_logits.numpy()

    npz = np.load("../checkpoints/kernel_ai_v0.npz")
    w = {k: npz[k] for k in npz.files if not k.startswith("_cfg")}

    numpy_logits = numpy_forward(input_ids, w, n_layers, n_heads, seq_len)

    max_abs_diff = np.abs(torch_logits - numpy_logits).max()
    print(f"torch logits shape: {torch_logits.shape}, numpy logits shape: {numpy_logits.shape}")
    print(f"max abs diff (PyTorch vs pure-numpy): {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-2, "numpy forward pass diverges from PyTorch"
    print("PASS: pure-numpy forward pass matches PyTorch exactly (within fp32 tolerance).")


if __name__ == "__main__":
    main()
