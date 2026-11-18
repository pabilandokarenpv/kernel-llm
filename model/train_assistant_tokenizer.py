"""
Train the Kernel AI tokenizer — v0.1, general-purpose vocab.

Unlike the interpretability project's tokenizer (trained only on synthetic
Kernel programs), this one is trained on real natural-language + instruction
data (TinyStories + Dolly15k), because Kernel AI needs to actually produce
English, not just code. Still 100% our own tokenizer, no reused GPT-2/Llama
vocab.
"""

import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train(corpus_path, out_path, vocab_size=16384):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<|endoftext|>"],
        min_frequency=2,
    )

    def iter_lines():
        with open(corpus_path) as f:
            buf = []
            for line in f:
                buf.append(line)
                if len(buf) >= 2000:
                    yield "".join(buf)
                    buf = []
            if buf:
                yield "".join(buf)

    tokenizer.train_from_iterator(iter_lines(), trainer=trainer)
    tokenizer.save(out_path)
    print(f"saved tokenizer to {out_path}, vocab size {tokenizer.get_vocab_size()}")

    sample = "Once upon a time, a mole and a little girl became best friends."
    ids = tokenizer.encode(sample).ids
    decoded = tokenizer.decode(ids)
    print(f"sample: {len(sample)} chars -> {len(ids)} tokens")
    print("round-trip matches:", decoded.strip() == sample.strip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/assistant_corpus.txt")
    ap.add_argument("--out", default="../data/assistant_tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=16384)
    args = ap.parse_args()
    train(args.corpus, args.out, args.vocab_size)
