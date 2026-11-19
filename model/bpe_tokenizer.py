"""
Pure-Python GPT-2-style byte-level BPE tokenizer — encode/decode only,
reading a HuggingFace `tokenizers`-format tokenizer.json directly.

Exists because the real `tokenizers` library is Rust-backed (a native
extension) and Cloudflare's Pyodide-based Python Workers can't load
native-extension packages the same way they can pure-Python/numpy ones
(the same class of problem that blocked onnxruntime-web — see
architecture/ARDs/ARD-003-serving-infrastructure.md). This reimplements
just the byte-level BPE algorithm in stdlib-only Python so it runs
anywhere, including inside a Worker.
"""

import json
import re

# GPT-2's byte-to-unicode mapping: maps every possible byte value (0-255)
# to a printable unicode character, so BPE merges can operate on text
# without worrying about non-printable bytes. Standard, from the original
# GPT-2 BPE implementation.
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}

# GPT-2's pre-tokenizer regex (splits text into chunks BPE merges run on).
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\s\w]+| ?\w+| ?\d+|\s+(?!\S)|\s+"""
)


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class ByteLevelBPE:
    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path) as f:
            tok = json.load(f)
        model = tok["model"]
        self.vocab = model["vocab"]  # str -> id
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        merges = model["merges"]
        # merges can be list of [a, b] or list of "a b" strings depending on version
        self.bpe_ranks = {}
        for i, m in enumerate(merges):
            if isinstance(m, str):
                a, b = m.split(" ", 1)
            else:
                a, b = m[0], m[1]
            self.bpe_ranks[(a, b)] = i
        self.cache = {}

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            return token
        while True:
            min_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if min_pair not in self.bpe_ranks:
                break
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        ids = []
        for chunk in _PAT.findall(text):
            byte_encoded = "".join(_BYTE_TO_UNICODE[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(byte_encoded).split(" "):
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                else:
                    ids.append(self.vocab.get("<unk>", 1))
        return ids

    def decode(self, ids):
        text = "".join(self.id_to_token.get(i, "") for i in ids)
        byte_arr = bytearray(_UNICODE_TO_BYTE.get(c, 0) for c in text)
        return byte_arr.decode("utf-8", errors="replace")
