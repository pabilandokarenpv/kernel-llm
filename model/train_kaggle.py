"""
The REAL training script — meant to run on Kaggle's free GPU/TPU (30
GPU-hrs/week + 20 TPU-hrs/week, see kernel-lang's LLM roadmap research for
verified current limits), not locally. This script is genuine, runnable
code, but actually executing a full run needs your Kaggle account; that's
a manual step, not something driven headlessly from here.

Usage on Kaggle:
  1. Upload this repo (or `git clone` it in a notebook cell).
  2. Regenerate the corpus (not committed to git, it's 90MB+):
       python corpus/build_corpus.py --n 20000 --out data/corpus.jsonl \
           --kernel-bin <path to the built `kernel` binary from kernel-lang>
  3. Train the tokenizer: python model/train_tokenizer.py
  4. Run this script: python model/train_kaggle.py --size 40M --steps 5000

Logs to Weights & Biases if `wandb` is installed and you're logged in
(free tier, no time limit on tracked hours) — optional, never required.
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import KernelLLM, KernelLLMConfig
from muon import Muon, split_muon_adamw_params
from tokenizers import Tokenizer

SIZE_PRESETS = {
    # (d_model, n_layers, n_heads) — the 3-point scaling-law grid from the
    # roadmap: train all three on identical data/steps to plot loss vs
    # non-embedding params, even 3 points is a real figure if the fit is
    # shown honestly.
    "10M": dict(d_model=256, n_layers=4, n_heads=4),
    "40M": dict(d_model=384, n_layers=8, n_heads=6),
    "125M": dict(d_model=768, n_layers=10, n_heads=12),
}


def load_data(corpus_path, tokenizer_path, seq_len, val_fraction=0.05):
    tok = Tokenizer.from_file(tokenizer_path)
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    ids = []
    with open(corpus_path) as f:
        for line in f:
            rec = json.loads(line)
            ids.append(bos)
            ids.extend(tok.encode(rec["source"]).ids)
            ids.append(eos)
    data = torch.tensor(ids, dtype=torch.long)
    n_chunks = len(data) // seq_len
    data = data[: n_chunks * seq_len].view(n_chunks, seq_len)
    n_val = max(1, int(n_chunks * val_fraction))
    return data[n_val:], data[:n_val], tok.get_vocab_size()


def get_batch(data, batch_size, device):
    idx = torch.randint(0, data.shape[0], (batch_size,))
    batch = data[idx].to(device)
    return batch[:, :-1], batch[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=SIZE_PRESETS.keys(), default="40M")
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(__file__), "../data/corpus.jsonl"))
    ap.add_argument("--tokenizer", default=os.path.join(os.path.dirname(__file__), "../data/tokenizer.json"))
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-adamw", type=float, default=3e-4)
    ap.add_argument("--use-moe", action="store_true", default=True)
    ap.add_argument("--n-experts", type=int, default=6)
    ap.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(__file__), "../checkpoints"))
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--dry-run", action="store_true",
                     help="tiny config + few steps, for local CPU correctness checks before spending Kaggle GPU-hours")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_data, val_data, vocab_size = load_data(args.corpus, args.tokenizer, args.seq_len)
    print(f"train sequences: {train_data.shape[0]}, val: {val_data.shape[0]}, vocab: {vocab_size}")

    if args.dry_run:
        preset = dict(d_model=64, n_layers=2, n_heads=2)
        args.n_experts = 2
        args.steps = min(args.steps, 5)
        args.checkpoint_every = 10**9  # skip checkpointing in a dry run
        print("DRY RUN: tiny config, few steps, correctness check only")
    else:
        preset = SIZE_PRESETS[args.size]
    cfg = KernelLLMConfig(
        vocab_size=vocab_size, max_seq_len=args.seq_len,
        use_moe=args.use_moe, n_experts=args.n_experts, top_k=1,
        **preset,
    )
    model = KernelLLM(cfg).to(device)
    print(f"model: {args.size}, {model.num_params():,} params")

    muon_params, adamw_params = split_muon_adamw_params(model)
    muon_opt = Muon(muon_params, lr=args.lr_muon)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=args.lr_adamw)

    wandb = None
    if args.wandb_project:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("wandb not installed, skipping logging (pip install wandb to enable)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        x, y = get_batch(train_data, args.batch_size, device)
        _, loss = model(x, y)
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        loss.backward()
        muon_opt.step()
        adamw_opt.step()

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, min(args.batch_size, val_data.shape[0]), device)
                _, val_loss = model(vx, vy)
            elapsed = time.time() - t0
            print(f"step {step:5d}  train_loss {loss.item():.4f}  val_loss {val_loss.item():.4f}  "
                  f"elapsed {elapsed:.0f}s")
            if wandb:
                wandb.log({"train_loss": loss.item(), "val_loss": val_loss.item(), "step": step})

        if step > 0 and step % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"kernel_llm_{args.size}_step{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step}, ckpt_path)
            print(f"  checkpoint saved: {ckpt_path}")

    final_path = os.path.join(args.checkpoint_dir, f"kernel_llm_{args.size}_final.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": args.steps}, final_path)
    print(f"training complete, final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
