"""
Kernel AI inference Space — pure numpy forward pass, real from-scratch
checkpoint (no hosted model ever answers here, matches ARD-002's rule).

Why this exists: Cloudflare Workers Free caps CPU time at 10ms/request,
confirmed too small for any real transformer forward pass (ARD-003). HF
Spaces' free CPU Basic tier runs a persistent container instead of a
per-request CPU budget, so the same numpy_forward.py / bpe_tokenizer.py
code (already verified byte-for-byte against the real PyTorch model)
just runs here without a CPU-time wall.

UI: a custom gr.Blocks layout (not the stock ChatInterface look) styled
to be minimal/dark like Claude or ChatGPT — Gradio's own chrome (footer,
settings gear, "Use via API", the sliders panel) is hidden via CSS so the
only thing visible is the chat itself.
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


EOS_ID = 4  # <|endoftext|> -- the document separator every training source used;
            # without stopping on it, generation drifts into the NEXT document's
            # leaked template text (confirmed real bug: a user got back
            # "### Instruction:\nWhat are some ways..." garbage from an
            # unrelated training example, not a hallucination, an un-stopped
            # generation running straight through a document boundary)


def generate_reply(message, history, max_new_tokens, temperature):
    # The training corpus is entirely "### Instruction:\n{msg}\n\n### Response:\n{reply}"
    # (see data/build_final_corpus.py) -- prompting in that exact format,
    # not the raw message, is what the model actually learned to complete
    # coherently; feeding it bare text was a prompt/training-format mismatch.
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    ids = tok.encode(prompt)
    input_ids = np.array([ids], dtype=np.int64)
    rng = np.random.default_rng()
    t0 = time.time()
    out_ids = generate(
        input_ids, w, N_LAYERS, N_HEADS, SEQ_LEN,
        max_new_tokens=int(max_new_tokens), temperature=float(temperature),
        rng=rng, eos_id=EOS_ID,
    )
    elapsed = time.time() - t0
    completion_ids = out_ids[len(ids):]
    if completion_ids and completion_ids[-1] == EOS_ID:
        completion_ids = completion_ids[:-1]
    text = tok.decode(completion_ids).strip()
    n_new = len(completion_ids)
    return text, elapsed, n_new


def generate_reply_html(message, history, max_new_tokens, temperature):
    """Gradio UI variant: text + a small HTML meta footer."""
    text, elapsed, n_new = generate_reply(message, history, max_new_tokens, temperature)
    meta = f"\n\n<span class='gen-meta'>{elapsed:.1f}s · {n_new} tokens</span>"
    return text + meta


CSS = """
:root {
  --bg: #212121; --bg-elevated: #2a2a2a; --bubble-user: #2f2f2f;
  --border: #3a3a3a; --text: #ececec; --text-muted: #8e8ea0;
}
html, body { background: var(--bg) !important; }
* { box-sizing: border-box; }
.gradio-container, .gradio-container .main, gradio-app {
  background: var(--bg) !important; max-width: 100% !important;
  min-height: 100vh !important; font-family: -apple-system, BlinkMacSystemFont,
  "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
footer, .built-with, #gradio-api-pill, .settings, .show-api { display: none !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

#app-shell { max-width: 820px; margin: 0 auto; width: 100%; padding: 0 20px; }

#header-row { position: sticky; top: 0; z-index: 20; background: var(--bg);
  padding: 22px 0 14px 0; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.06); }
#header-row h1 { font-size: 1.1rem; font-weight: 600; color: var(--text);
  letter-spacing: -0.01em; margin: 0 0 4px 0; }
#header-row p { font-size: 0.8rem; color: var(--text-muted); margin: 0;
  line-height: 1.5; max-width: 560px; margin: 0 auto; }

#chatbox { background: transparent !important; border: none !important;
  padding-top: 8px !important; }
#chatbox .message-wrap { background: transparent !important; }
.message.user { background: var(--bubble-user) !important; border-radius: 20px !important;
  color: var(--text) !important; border: none !important; padding: 10px 16px !important;
  animation: fadeIn 0.25s ease; }
.message.bot, .message.bot.panel-full-width { background: transparent !important;
  color: #d8d8d8 !important; border: none !important; box-shadow: none !important;
  line-height: 1.65 !important; animation: fadeIn 0.25s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
.gen-meta { font-size: 0.7rem; color: #6a6a6a; }

#advanced-accordion { border: none !important; background: transparent !important; }
#advanced-accordion .label-wrap { color: var(--text-muted) !important; font-size: 0.75rem !important; }

#input-dock { position: sticky; bottom: 0; background: var(--bg);
  padding: 10px 0 calc(16px + env(safe-area-inset-bottom)) 0; }
#input-row textarea { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
  border-radius: 24px !important; color: var(--text) !important; font-size: 1rem !important; }
#input-row textarea::placeholder { color: var(--text-muted) !important; }
button.primary { background: var(--text) !important; color: var(--bg) !important;
  border-radius: 20px !important; border: none !important; min-width: 44px !important; }

/* Tablet */
@media (max-width: 900px) {
  #app-shell { padding: 0 14px; }
}
/* Mobile */
@media (max-width: 600px) {
  #app-shell { padding: 0 10px; }
  #header-row { padding: 14px 0 10px 0; }
  #header-row h1 { font-size: 1rem; }
  #header-row p { font-size: 0.72rem; }
  .message.user, .message.bot { font-size: 0.95rem !important; }
  #input-row textarea { font-size: 16px !important; } /* 16px avoids iOS auto-zoom on focus */
  button.primary { min-width: 40px !important; }
}
"""

with gr.Blocks(title="Kernel AI") as demo:
    with gr.Column(elem_id="app-shell"):
        with gr.Column(elem_id="header-row"):
            gr.Markdown(
                "# Kernel AI\n"
                "<p>A genuinely from-scratch LLM — 33M params, trained from random init, "
                "no hosted model behind this. Small and honest: coherent short prose, "
                "structurally correct code/math, not always factually correct.</p>"
            )

        chatbot = gr.Chatbot(elem_id="chatbox", height="65vh", show_label=False,
                              avatar_images=(None, None), buttons=["copy"], layout="bubble")

        with gr.Accordion("Advanced", open=False, elem_id="advanced-accordion"):
            max_new_tokens = gr.Slider(1, 64, value=30, step=1, label="max_new_tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")

        with gr.Column(elem_id="input-dock"):
            with gr.Row(elem_id="input-row"):
                msg = gr.Textbox(placeholder="Message Kernel AI...", show_label=False, scale=9,
                                  container=False, autofocus=True)
                send = gr.Button("↑", scale=1, variant="primary")

    def user_turn(message, history):
        return "", history + [{"role": "user", "content": message}]

    def bot_turn(history, max_new_tokens, temperature):
        last = history[-1]
        user_message = last["content"] if isinstance(last, dict) else last.content
        if isinstance(user_message, list):
            # Gradio 6 can wrap content as a list of parts; flatten to text.
            user_message = "".join(
                p if isinstance(p, str) else p.get("text", "") for p in user_message
            )
        reply = generate_reply_html(user_message, history, max_new_tokens, temperature)
        history.append({"role": "assistant", "content": reply})
        return history

    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, max_new_tokens, temperature], [chatbot]
    )
    send.click(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, max_new_tokens, temperature], [chatbot]
    )

if __name__ == "__main__":
    import fastapi
    import uvicorn

    api = fastapi.FastAPI()

    @api.post("/api/generate")
    async def api_generate(req: fastapi.Request):
        """Clean JSON API for other frontends (Slate) to call directly,
        alongside this file's own Gradio UI — same model, same weights,
        just a plain request/response contract instead of Gradio's
        call/poll protocol. {"messages": [{"role", "content"}, ...]} in,
        {"text": "..."} out."""
        body = await req.json()
        messages = body.get("messages", [])
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        mnt = int(body.get("max_new_tokens", 60))
        temp = float(body.get("temperature", 0.8))
        text, elapsed, n_new = generate_reply(last_user, [], mnt, temp)
        return {"text": text, "elapsed_s": round(elapsed, 2), "tokens": n_new}

    app = gr.mount_gradio_app(api, demo, path="/", css=CSS,
                               theme=gr.themes.Base(primary_hue="neutral", neutral_hue="neutral"))
    uvicorn.run(app, host="0.0.0.0", port=7860)
