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


def generate_reply(message, history, max_new_tokens, temperature):
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
    meta = f"\n\n<span class='gen-meta'>{elapsed:.1f}s · {len(out_ids) - len(ids)} tokens</span>"
    return text + meta


CSS = """
.gradio-container { max-width: 760px !important; margin: 0 auto !important;
    background: #1a1a1a !important; }
footer, .built-with, #gradio-api-pill, .settings { display: none !important; }
#header-row { padding: 28px 8px 4px 8px; text-align: center; }
#header-row h1 { font-size: 1.15rem; font-weight: 600; color: #ececec;
    letter-spacing: -0.01em; margin: 0 0 4px 0; }
#header-row p { font-size: 0.8rem; color: #8a8a8a; margin: 0; line-height: 1.5; }
#chatbox { background: transparent !important; border: none !important; }
#chatbox .message-wrap { background: transparent !important; }
.message.user { background: #2f2f2f !important; border-radius: 18px !important;
    color: #ececec !important; border: none !important; }
.message.bot, .message.bot.panel-full-width { background: transparent !important;
    color: #d8d8d8 !important; border: none !important; box-shadow: none !important; }
.gen-meta { font-size: 0.7rem; color: #6a6a6a; }
#input-row { max-width: 760px; margin: 0 auto; padding: 0 8px 20px 8px; }
#input-row textarea { background: #2b2b2b !important; border: 1px solid #3a3a3a !important;
    border-radius: 24px !important; color: #ececec !important; }
#advanced-accordion { border: none !important; background: transparent !important; }
#advanced-accordion .label-wrap { color: #6a6a6a !important; font-size: 0.75rem !important; }
button.primary { background: #ececec !important; color: #1a1a1a !important;
    border-radius: 20px !important; border: none !important; }
"""

with gr.Blocks(title="Kernel AI") as demo:
    with gr.Column(elem_id="header-row"):
        gr.Markdown(
            "# Kernel AI\n"
            "<p>A genuinely from-scratch LLM — 33M params, trained from random init, "
            "no hosted model behind this. Small and honest: coherent short prose, "
            "structurally correct code/math, not always factually correct.</p>"
        )

    chatbot = gr.Chatbot(elem_id="chatbox", height=480, show_label=False,
                          avatar_images=(None, None), buttons=["copy"], layout="bubble")

    with gr.Accordion("Advanced", open=False, elem_id="advanced-accordion"):
        max_new_tokens = gr.Slider(1, 64, value=30, step=1, label="max_new_tokens")
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")

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
        reply = generate_reply(user_message, history, max_new_tokens, temperature)
        history.append({"role": "assistant", "content": reply})
        return history

    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, max_new_tokens, temperature], [chatbot]
    )
    send.click(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot, max_new_tokens, temperature], [chatbot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                 css=CSS, theme=gr.themes.Base(primary_hue="neutral", neutral_hue="neutral"))
