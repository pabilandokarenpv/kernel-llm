"""
Smoke test for the Kernel AI assistant model — proves the pipeline works on
REAL natural-language + instruction data (not just the narrow Kernel-corpus
interpretability model). Small, CPU, a few hundred steps — enough to prove
loss decreases and to see real (if weak) generated text, not to produce a
good model. The real training run still needs Kaggle GPU hours.
"""

import json
import sys
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import KernelLLM, KernelLLMConfig


def load_tokens(corpus_path, tokenizer_path, seq_len=256, max_chars=8_000_000):
    tok = Tokenizer.from_file(tokenizer_path)
    with open(corpus_path) as f:
        text = f.read(max_chars)
    ids = tok.encode(text).ids
    data = torch.tensor(ids, dtype=torch.long)
    n_chunks = len(data) // seq_len
    data = data[: n_chunks * seq_len].view(n_chunks, seq_len)
    return data, tok


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=60, device="cpu"):
    model.eval()
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -model.cfg.max_seq_len :])
        next_id = torch.argmax(logits[0, -1]).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
    return tok.decode(x[0].tolist())


def main():
    here = os.path.dirname(__file__)
    corpus_path = os.path.join(here, "../data/assistant_corpus.txt")
    tokenizer_path = os.path.join(here, "../data/assistant_tokenizer.json")

    seq_len = 128
    data, tok = load_tokens(corpus_path, tokenizer_path, seq_len=seq_len, max_chars=3_000_000)
    print(f"loaded {data.shape[0]} sequences of length {seq_len}, vocab_size={tok.get_vocab_size()}")

    torch.manual_seed(0)
    cfg = KernelLLMConfig(vocab_size=tok.get_vocab_size(), d_model=192, n_layers=3, n_heads=4,
                           max_seq_len=seq_len, use_moe=False)
    model = KernelLLM(cfg)
    print(f"model params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch_size = 8
    n_steps = 80
    losses = []
    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, data.shape[0], (batch_size,))
        batch = data[idx]
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 25 == 0 or step == n_steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    first_avg = sum(losses[:5]) / 5
    last_avg = sum(losses[-5:]) / 5
    print(f"\nfirst-5-step avg loss: {first_avg:.4f}")
    print(f"last-5-step avg loss:  {last_avg:.4f}")
    assert last_avg < first_avg, "loss did not decrease"
    print("PASS: loss decreased on REAL natural-language + instruction data.\n")

    print("--- sample generation (greedy, 150 steps of training, honest output) ---")
    for prompt in ["Once upon a time,", "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"]:
        out = generate(model, tok, prompt, max_new_tokens=40)
        print(f"PROMPT: {prompt!r}")
        print(f"OUTPUT: {out!r}\n")

    ckpt_dir = os.path.join(here, "../checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "kernel_ai_smoke_v0.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, ckpt_path)
    print(f"checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
