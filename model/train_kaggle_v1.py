"""
Kernel AI — the REAL training run, meant to execute on Kaggle's free T4x2
GPU. Pushed and monitored via the Kaggle API (kaggle kernels push/status/
output), not a notebook anyone has to click through manually.

Reads the final merged corpus + tokenizer from a Kaggle Dataset
(karenpearlvpabilando/kernel-ai-corpus) mounted at /kaggle/input, avoiding
GitHub's size limits entirely (the raw corpus is 98MB).

Dense SwiGLU architecture (no MoE) — matches the pure-numpy serving path
already proven in the Python Worker, so this checkpoint is directly
servable once training is done, no architecture mismatch to debug later.
"""

import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import KernelLLM, KernelLLMConfig
from muon import Muon, split_muon_adamw_params

# Training data comes from a Kaggle Dataset (karenpearlvpabilando/kernel-ai-corpus),
# mounted read-only at this path when the dataset is attached to the kernel —
# not fetched over the network, so no GitHub size limits apply.
DATASET_DIR = "/kaggle/input/kernel-ai-corpus"

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# Sized for a real multi-hour Kaggle T4x2 session: enough capacity to be
# meaningfully more than a toy, small enough to actually finish real steps
# in a single session instead of timing out mid-run.
D_MODEL, N_LAYERS, N_HEADS = 384, 8, 6
SEQ_LEN = 256
BATCH_SIZE = 24
# 4x the previous run (6000 -> 24000): same architecture/corpus size (~98MB,
# ~0.5s/step observed last run) so this is ~3.3hrs wall clock on a T4,
# comfortably inside one Kaggle session -- more passes over a corpus that's
# now also better-balanced (less TinyStories, real human chit-chat added)
# should move val_loss meaningfully past the 2.27 the last run reached,
# since loss was still dropping at step 6000, not plateaued.
TOTAL_STEPS = 24000
CHECKPOINT_EVERY = 2000
LR_MUON = 0.02
LR_ADAMW = 3e-4


def load_data(corpus_path, tokenizer_path, seq_len, val_fraction=0.02, max_chars=None):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    EOS_ID = tok.token_to_id("<|endoftext|>")
    assert EOS_ID is not None, "<|endoftext|> missing from tokenizer vocab"
    with open(corpus_path, "r", errors="ignore") as f:
        text = f.read(max_chars) if max_chars else f.read()
    print(f"  corpus text: {len(text)/1e6:.1f}M chars")
    # Encode in per-document batches (split on the corpus's own
    # <|endoftext|> delimiters) via encode_batch, not one tok.encode() call
    # on the whole multi-hundred-MB string -- the monolithic call's peak
    # memory during pretokenization scales badly and got OOM-killed by
    # Kaggle on a 280MB corpus (v4) even though the final token array
    # itself is tiny ("Killed" in the kernel log, no Python traceback --
    # that signature is the Linux OOM killer). Verified on a real 20MB
    # slice of the actual corpus: identical token count to the old
    # monolithic method, and 5x faster (25.1s -> 5.1s for that slice).
    chunks = [c for c in text.split("<|endoftext|>") if c.strip()]
    del text
    ids = []
    BATCH = 2000
    for i in range(0, len(chunks), BATCH):
        for e in tok.encode_batch(chunks[i : i + BATCH]):
            ids.extend(e.ids)
            ids.append(EOS_ID)
        if i % (BATCH * 5) == 0:
            print(f"  tokenized {i}/{len(chunks)} chunks...")
    print(f"  tokenized: {len(ids)/1e6:.1f}M tokens, vocab={tok.get_vocab_size()}")
    data = torch.tensor(ids, dtype=torch.long)
    n_chunks = len(data) // seq_len
    data = data[: n_chunks * seq_len].view(n_chunks, seq_len)
    n_val = max(1, int(n_chunks * val_fraction))
    return data[n_val:], data[:n_val], tok.get_vocab_size()


def get_batch(data, batch_size, device):
    idx = torch.randint(0, data.shape[0], (batch_size,))
    batch = data[idx].to(device)
    return batch[:, :-1], batch[:, 1:]


def find_dataset_file(filename):
    """Kaggle's dataset mount path isn't always a flat /kaggle/input/<slug>/
    — walk the tree to find the file wherever it actually landed."""
    for root_dir in ("/kaggle/input", DATASET_DIR):
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _, filenames in os.walk(root_dir):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"torch version: {torch.__version__}, cuda built: {torch.version.cuda}")

    if os.path.isdir("/kaggle/input"):
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            print(f"  {dirpath}: dirs={dirnames} files={filenames[:5]}")
    else:
        print("NO /kaggle/input DIR")

    corpus_path = find_dataset_file("kernel_ai_final_corpus.txt")
    tokenizer_path = find_dataset_file("kernel_ai_final_tokenizer.json")
    if not corpus_path or not tokenizer_path:
        raise FileNotFoundError(
            "could not locate kernel_ai_final_corpus.txt / kernel_ai_final_tokenizer.json "
            "anywhere under /kaggle/input — attach the 'kernel-ai-corpus' dataset to this kernel."
        )

    train_data, val_data, vocab_size = load_data(corpus_path, tokenizer_path, SEQ_LEN)
    print(f"train sequences: {train_data.shape[0]}, val: {val_data.shape[0]}")

    cfg = KernelLLMConfig(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        max_seq_len=SEQ_LEN, use_moe=False,
    )
    model = KernelLLM(cfg).to(device)
    print(f"model: {model.num_params():,} params")

    muon_params, adamw_params = split_muon_adamw_params(model)
    muon_opt = Muon(muon_params, lr=LR_MUON)
    adamw_opt = torch.optim.AdamW(adamw_params, lr=LR_ADAMW)

    log_path = os.path.join(OUT_DIR, "train_log.jsonl")
    t0 = time.time()
    for step in range(TOTAL_STEPS):
        model.train()
        x, y = get_batch(train_data, BATCH_SIZE, device)
        _, loss = model(x, y)
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        loss.backward()
        muon_opt.step()
        adamw_opt.step()

        if step % 50 == 0 or step == TOTAL_STEPS - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, min(BATCH_SIZE, val_data.shape[0]), device)
                _, val_loss = model(vx, vy)
            elapsed = time.time() - t0
            line = f"step {step:5d}  train_loss {loss.item():.4f}  val_loss {val_loss.item():.4f}  elapsed {elapsed:.0f}s"
            print(line)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "train_loss": loss.item(),
                                     "val_loss": val_loss.item(), "elapsed_s": elapsed}) + "\n")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(OUT_DIR, f"kernel_ai_v1_step{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": step}, ckpt_path)
            print(f"  checkpoint saved: {ckpt_path}")

    final_path = os.path.join(OUT_DIR, "kernel_ai_v1_final.pt")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "step": TOTAL_STEPS}, final_path)
    print(f"training complete, final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
