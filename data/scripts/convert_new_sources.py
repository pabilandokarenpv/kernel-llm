"""
Converts two new free, human-authored HuggingFace datasets into this
project's {instruction, response} JSONL format, matching the existing
raw/*.jsonl sources so build_final_corpus.py can pick them up unchanged.

- HuggingFaceH4/no_robots: ~10k human-WRITTEN instructions (the name is
  literal -- this dataset exists specifically because it's NOT model-
  generated). Consecutive user/assistant turns -> one pair each.
- AlekseyKorshuk/persona-chat (ConvAI2 PersonaChat): human crowdsourced
  persona-grounded chit-chat. Every adjacent turn in each conversation's
  history becomes one pair -- this is what's missing from the existing
  mix (Dolly/OASST2/CodeAlpaca/GSM8K are all task-oriented; nothing
  currently teaches ordinary two-person small talk, which is exactly
  the "hi" -> unrelated-nonsense failure mode this was pulled in to fix).

Both are genuinely free and human-authored -- no hosted/existing AI
model was used to generate any of this data, per the explicit
instruction not to.
"""
import json
import os

import pyarrow.parquet as pq

BASE = os.path.join(os.path.dirname(__file__), "..", "raw")


def convert_no_robots():
    t = pq.read_table(os.path.join(BASE, "no_robots.parquet"))
    rows = t.to_pylist()
    out = []
    for row in rows:
        msgs = row["messages"]
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                instr = msgs[i]["content"].strip()
                resp = msgs[i + 1]["content"].strip()
                if instr and resp and len(instr) < 2000 and len(resp) < 2000:
                    out.append({"instruction": instr, "response": resp})
    path = os.path.join(BASE, "no_robots_pairs.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"no_robots: {len(out)} pairs -> {path}")


def convert_personachat():
    t = pq.read_table(os.path.join(BASE, "personachat.parquet"))
    rows = t.to_pylist()
    out = []
    for row in rows:
        history = None
        for u in row["utterances"]:
            history = u["history"]  # grows each turn; last one has the full conversation
        if not history or len(history) < 2:
            continue
        for i in range(len(history) - 1):
            instr = history[i].strip()
            resp = history[i + 1].strip()
            if instr and resp and len(instr) < 500 and len(resp) < 500:
                out.append({"instruction": instr, "response": resp})
    path = os.path.join(BASE, "personachat_pairs.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"personachat: {len(out)} pairs -> {path}")


if __name__ == "__main__":
    convert_no_robots()
    convert_personachat()
