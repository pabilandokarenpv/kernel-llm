"""
Kernel AI inference Space — pure numpy forward pass, real from-scratch
checkpoint (no hosted model ever answers here, matches ARD-002's rule).

Why this exists: Cloudflare Workers Free caps CPU time at 10ms/request,
confirmed too small for any real transformer forward pass (ARD-003). HF
Spaces' free CPU Basic tier runs a persistent container instead of a
per-request CPU budget, so the same numpy_forward.py / bpe_tokenizer.py
code (already verified byte-for-byte against the real PyTorch model)
just runs here without a CPU-time wall.
"""
import json
import time

import gradio as gr
import numpy as np

from numpy_forward import generate
from bpe_tokenizer import ByteLevelBPE

CKPT_PATH = "kernel_ai_v1.raw"
TOKENIZER_PATH = "kernel_ai_final_tokenizer.json"
N_LAYERS, N_HEADS, SEQ_LEN = 8, 6, 256

print("loading tokenizer...")
tok = ByteLevelBPE(TOKENIZER_PATH)

print("loading checkpoint...")
t0 = time.time()
with open(CKPT_PATH, "rb") as f:
    data = f.read()
mlen = int.from_bytes(data[:4], "little")
manifest = json.loads(data[4 : 4 + mlen])
payload = data[4 + mlen :]
w = {}
for k, info in manifest.items():
    if k == "_cfg":
        continue
    dtype = np.float16 if info.get("dtype") == "float16" else np.float32
    itemsize = 2 if dtype is np.float16 else 4
    arr = np.frombuffer(
        payload, dtype=dtype, count=info["nbytes"] // itemsize, offset=info["offset"]
    ).reshape(info["shape"]).astype(np.float32)
    w[k] = arr
print(f"checkpoint loaded in {time.time() - t0:.2f}s, {len(w)} arrays")


def respond(message, history, max_new_tokens, temperature):
    ids = tok.encode(message)
    input_ids = np.array([ids], dtype=np.int64)
    rng = np.random.default_rng()
    t0 = time.time()
    out_ids = generate(
        input_ids, w, N_LAYERS, N_HEADS, SEQ_LEN,
        max_new_tokens=int(max_new_tokens), temperature=float(temperature), rng=rng,
    )
    elapsed = time.time() - t0
    text = tok.decode(out_ids)
    return f"{text}\n\n_({elapsed:.1f}s, {len(out_ids) - len(ids)} tokens generated)_"


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Slider(1, 64, value=30, step=1, label="max_new_tokens"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature"),
    ],
    title="Kernel AI — genuinely from-scratch LLM",
    description=(
        "33M params, trained from random init on a 110,187-example corpus "
        "(TinyStories + OASST2 + CodeAlpaca + Dolly15k + GSM8K + Kernel-lang "
        "traces), 6000 steps on a free Kaggle T4, val_loss 10.08 → 2.27. "
        "Honest expectations: small model, coherent short prose, structurally "
        "correct code/math formatting, not always factually/numerically correct."
    ),
)

if __name__ == "__main__":
    demo.launch()
