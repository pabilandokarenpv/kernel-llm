"""
Smoke test: proves the model + tokenizer + data pipeline actually works,
on CPU, with a tiny config. This is NOT the real training run — that
happens on Kaggle free GPU/TPU per the roadmap. This just proves the code
is correct (loss decreases on real tokenized Kernel programs) before
spending any cloud GPU-hours on it.
"""

import json
import sys
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import KernelLLM, KernelLLMConfig


def load_corpus_tokens(corpus_path, tokenizer_path, max_programs=200, seq_len=256):
    tok = Tokenizer.from_file(tokenizer_path)
    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    all_ids = []
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if i >= max_programs:
                break
            rec = json.loads(line)
            ids = tok.encode(rec["source"]).ids
            all_ids.append(bos)
            all_ids.extend(ids)
            all_ids.append(eos)
    data = torch.tensor(all_ids, dtype=torch.long)
    n_chunks = len(data) // seq_len
    data = data[: n_chunks * seq_len].view(n_chunks, seq_len)
    return data, tok.get_vocab_size()


def main():
    here = os.path.dirname(__file__)
    corpus_path = os.path.join(here, "../data/corpus.jsonl")
    tokenizer_path = os.path.join(here, "../data/tokenizer.json")

    seq_len = 128
    data, vocab_size = load_corpus_tokens(corpus_path, tokenizer_path, max_programs=300, seq_len=seq_len)
    print(f"loaded {data.shape[0]} sequences of length {seq_len}, vocab_size={vocab_size}")

    torch.manual_seed(0)
    cfg = KernelLLMConfig(vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4,
                           max_seq_len=seq_len, use_moe=True, n_experts=4, top_k=1)
    model = KernelLLM(cfg)
    print(f"model params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch_size = 8
    n_steps = 60
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
        if step % 10 == 0 or step == n_steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    first_avg = sum(losses[:5]) / 5
    last_avg = sum(losses[-5:]) / 5
    print(f"\nfirst-5-step avg loss: {first_avg:.4f}")
    print(f"last-5-step avg loss:  {last_avg:.4f}")
    assert last_avg < first_avg, "loss did not decrease — something is broken"
    print("PASS: loss decreased, model+tokenizer+data pipeline is correct.")


if __name__ == "__main__":
    main()
