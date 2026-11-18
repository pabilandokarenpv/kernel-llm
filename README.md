# Kernel-LLM

A from-scratch LLM interpretability research project: train a small
transformer on a language nobody's pretrained on (Kernel, `.kn`), then ask
whether it silently builds an internal model of program execution, variable
state, branch decisions, the same question Othello-GPT/Chess-GPT asked about
board games, applied to code instead.

The clean trick: since [Kernel](https://github.com/pabilandokarenpv/kernel-lang)
is a language we invented and its interpreter is hand-written (not scraped
from the internet), the corpus has **zero pretraining contamination** —
any "program understanding" the model shows is provably learned from this
corpus alone, something you can't claim about a model that's ever seen Python.

## Status: pipeline verified end to end, real training is next (needs your Kaggle account)

What's real and tested right now:
- **corpus generator** (`corpus/generate_programs.py`) — produces random,
  syntactically valid, checker-passing Kernel programs (int arithmetic,
  if/elif/else, for/while loops, bounded depth)
- **corpus builder** (`corpus/build_corpus.py`) — runs each program through
  the REAL Kernel interpreter (`kernel trace`) and keeps the ones that
  execute cleanly. Verified: 5000 random programs → 4829 succeeded (96.6%),
  failures are honestly counted integer-overflow panics in Rust debug-mode
  arithmetic from compounding loops, not corpus-builder bugs
- **tokenizer** (`model/train_tokenizer.py`) — custom BPE trained on the
  corpus (vocab 1195), verified exact round-trip on a real program
- **model** (`model/model.py`) — modernized nanoGPT: RoPE, RMSNorm, QK-Norm,
  SwiGLU, optional top-k MoE FFN with load-balancing aux loss, zero-init
  output head
- **Muon optimizer** (`model/muon.py`) — orthogonalized momentum SGD for
  hidden matrices (Jordan et al. 2024), AdamW for embeddings/norms/head
- **smoke test** (`model/smoke_test.py`) — proves the whole pipeline is
  correct: real tokenized corpus → real model → loss drops 7.11 → 2.04 over
  60 CPU steps
- **training script** (`model/train_kaggle.py`) — the actual training run,
  with a `--dry-run` flag verified working end to end (model forward,
  backward, Muon step, AdamW step, checkpoint save, all run clean)

What's NOT done, and genuinely can't be from here: the real training run.
Muon's Newton-Schulz iteration measured at ~16s/step on CPU (expected —
it's GPU-bound math, that's fine, the real run targets Kaggle's free T4/P100
or TPU v3-8). Running this at scale needs your Kaggle account, that's a
manual step for you, not something drivable headlessly.

## Reproducing locally

```bash
pip install -r requirements.txt

# 1. generate a corpus (needs the `kernel` binary built from kernel-lang)
python corpus/build_corpus.py --n 5000 --out data/corpus.jsonl \
    --kernel-bin ~/projects/kernel-lang/target/debug/kernel

# 2. train the tokenizer
python model/train_tokenizer.py --corpus data/corpus.jsonl --out data/tokenizer.json

# 3. smoke-test (CPU, ~1 min, proves the pipeline is correct)
python model/smoke_test.py

# 4. dry-run the real training script (tiny config, few steps, CPU-safe)
python model/train_kaggle.py --dry-run
```

## Running the real training on Kaggle

1. Upload/clone this repo + the built Kernel compiler into a Kaggle notebook
2. Regenerate a much larger corpus (10k-50k programs; not committed to git,
   ~90MB+ — host on [HF Hub datasets](https://huggingface.co/datasets) once
   at scale, matching the original research plan)
3. `python model/train_kaggle.py --size 10M --steps 5000` then `40M`, `125M`
   for the 3-point scaling-law grid, same data/steps-per-param ratio across
   all three
4. Log to Weights & Biases (`--wandb-project kernel-llm`, free tier)

## The actual research question (not yet run)

Once a real checkpoint exists: does the MoE router specialize by *program
structure* (loop depth, branch type) the way Othello-GPT's probes found
emergent board-state representations? Use `TransformerLens`-style linear
probes against the `kernel trace` ground-truth data (every `let`, every
branch decision, already captured per-program) — that's the whole point of
having `kernel trace` emit structured ground truth: it's the labeled data
for exactly this probing step.

## Repo layout

```
corpus/
  generate_programs.py   # random Kernel program generator
  build_corpus.py         # runs programs through `kernel trace`, builds JSONL
model/
  model.py                # the architecture
  muon.py                  # Muon optimizer
  train_tokenizer.py       # custom BPE tokenizer training
  smoke_test.py             # CPU correctness check (not real training)
  train_kaggle.py           # the real training script (run on Kaggle)
data/
  tokenizer.json           # committed (72KB) — corpus.jsonl is NOT (90MB+, regenerate it)
checkpoints/               # not committed — model weights go to HF Hub once real
```

## License

MIT
