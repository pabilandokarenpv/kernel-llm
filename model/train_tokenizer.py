"""
Train a custom BPE tokenizer on the Kernel (.kn) corpus.

A custom tokenizer is necessary here (can't reuse GPT-2's), the corpus is a
language nobody's pretrained on. Trained on the generated programs' SOURCE
only — the model predicts source code, traces are the held-out ground truth
used later for probing, not training input.
"""

import argparse
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def iter_sources(jsonl_path):
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            yield rec["source"]


def train(jsonl_path, out_path, vocab_size=4096):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(iter_sources(jsonl_path), trainer=trainer)
    tokenizer.save(out_path)
    print(f"saved tokenizer to {out_path}, vocab size {tokenizer.get_vocab_size()}")

    # sanity check: round-trip a real program
    sample = next(iter_sources(jsonl_path))
    ids = tokenizer.encode(sample).ids
    decoded = tokenizer.decode(ids)
    print(f"sample program: {len(sample)} chars -> {len(ids)} tokens "
          f"({len(sample) / len(ids):.2f} chars/token)")
    print("round-trip matches:", decoded.strip() == sample.strip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus.jsonl")
    ap.add_argument("--out", default="../data/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=4096)
    args = ap.parse_args()
    train(args.corpus, args.out, args.vocab_size)
