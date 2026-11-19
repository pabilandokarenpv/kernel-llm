"""
Kernel AI serving endpoint — Cloudflare Python Worker.

Real tokenize -> autoregressive generate -> detokenize pipeline, running
entirely on numpy (no PyTorch, no ONNX — see
architecture/ARDs/ARD-003-serving-infrastructure.md for why).

MODEL_URL / TOKENIZER_URL / the architecture constants below point at the
CURRENT placeholder checkpoint (80 CPU training steps, garbled output —
honestly a proof-of-pipeline, not a usable model yet). Swapping in the
real Kaggle-trained checkpoint once it lands is a one-step change: update
these constants to the new checkpoint's URL + dimensions, nothing else in
this file needs to change.
"""

from workers import Response

_cache = {}

MODEL_URL = "https://raw.githubusercontent.com/pabilandokarenpv/kernel-llm/main/checkpoints/kernel_ai_v1.raw"
TOKENIZER_URL = "https://raw.githubusercontent.com/pabilandokarenpv/kernel-llm/main/data/kernel_ai_final_tokenizer.json"

# Real checkpoint from the first full Kaggle T4 training run (6000 steps,
# val_loss 10.08 -> 2.27, ~46 min wall clock) -- not the earlier 80-CPU-step
# placeholder. Must match whatever MODEL_URL points at; swap all three
# together if a newer checkpoint replaces this one.
N_LAYERS, N_HEADS, SEQ_LEN = 8, 6, 256
MAX_NEW_TOKENS = 24  # kept small — generation is CPU-bound and Workers have a tight free CPU budget


async def get_weights():
    import json
    import numpy as np
    import httpx

    if "w" not in _cache:
        async with httpx.AsyncClient() as client:
            resp = await client.get(MODEL_URL, timeout=30.0)
            resp.raise_for_status()
            data = resp.content
            mlen = int.from_bytes(data[:4], "little")
            manifest = json.loads(data[4:4 + mlen])
            payload = data[4 + mlen:]
            w = {}
            for k, info in manifest.items():
                if k == "_cfg":
                    continue
                dtype = np.float16 if info.get("dtype") == "float16" else np.float32
                itemsize = 2 if dtype is np.float16 else 4
                arr = np.frombuffer(
                    payload, dtype=dtype, count=info["nbytes"] // itemsize, offset=info["offset"]
                ).reshape(info["shape"]).astype(np.float32)  # upcast once at load, matmuls run in fp32
                w[k] = arr
            _cache["w"] = w
    return _cache["w"]


async def get_tokenizer():
    import httpx
    from bpe_tokenizer import ByteLevelBPE

    if "tok" not in _cache:
        async with httpx.AsyncClient(headers={"Accept-Encoding": "identity"}) as client:
            resp = await client.get(TOKENIZER_URL, timeout=30.0)
            resp.raise_for_status()
            tok_path = "/tmp/_kernel_ai_tokenizer.json"
            with open(tok_path, "wb") as f:
                f.write(resp.content)
            _cache["tok"] = ByteLevelBPE(tok_path)
    return _cache["tok"]


async def on_fetch(request):
    try:
        import time
        import numpy as np
        from numpy_forward import generate

        if request.method == "OPTIONS":
            return Response("", headers={"Access-Control-Allow-Origin": "*",
                                          "Access-Control-Allow-Methods": "POST, OPTIONS",
                                          "Access-Control-Allow-Headers": "Content-Type"})

        body = await request.json() if request.method == "POST" else {}
        if hasattr(body, "to_py"):
            body = body.to_py()
        prompt = body.get("prompt", "Hello!") if isinstance(body, dict) else "Hello!"
        max_new = min(int(body.get("max_new_tokens", MAX_NEW_TOKENS)) if isinstance(body, dict) else MAX_NEW_TOKENS, 40)

        t0 = time.perf_counter()
        w = await get_weights()
        tok = await get_tokenizer()
        t1 = time.perf_counter()

        input_ids = tok.encode(prompt)
        if not input_ids:
            input_ids = [tok.vocab.get("<bos>", 2)]

        rng = np.random.default_rng()
        out_ids = generate(
            np.array([input_ids]), w, N_LAYERS, N_HEADS, SEQ_LEN,
            max_new_tokens=max_new, temperature=0.8, rng=rng,
        )
        t2 = time.perf_counter()

        generated_ids = out_ids[len(input_ids):]
        text = tok.decode(generated_ids)

        import json as _json
        return Response(
            _json.dumps({
                "response": text,
                "prompt_tokens": len(input_ids),
                "generated_tokens": len(generated_ids),
                "load_ms": round((t1 - t0) * 1000, 1),
                "generate_ms": round((t2 - t1) * 1000, 1),
                "model": "kernel-ai-v1-placeholder (pre-Kaggle-training checkpoint)",
            }),
            headers={"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        )
    except Exception as e:
        import traceback
        import json as _json
        return Response(
            _json.dumps({"error": str(e), "traceback": traceback.format_exc()}),
            status=500,
            headers={"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        )
