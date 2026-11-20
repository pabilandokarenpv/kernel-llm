"""
Converts MS MARCO v1.1 (microsoft/ms_marco, free, HF) into the
'search-then-answer' pattern: {instruction, search_query, search_results,
response}. This is real human data end to end -- `query` is an actual
Bing search query a person typed, `passages` are real retrieved web
passages, `answers` are human-written. Nothing here is AI-generated;
this is what teaches Whitekernel's model the search-tool pattern using
only real search-engine logs, not an existing AI model's outputs.
"""
import json
import os

import pyarrow.parquet as pq

BASE = os.path.join(os.path.dirname(__file__), "..", "raw")
NO_ANSWER = {"no answer present.", "no answer present", ""}


def convert():
    t = pq.read_table(os.path.join(BASE, "ms_marco.parquet"))
    rows = t.to_pylist()
    out = []
    for row in rows:
        query = (row.get("query") or "").strip()
        answers = row.get("answers") or []
        answer = (answers[0] if answers else "").strip()
        if not query or not answer or answer.lower() in NO_ANSWER:
            continue
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") or []
        selected = passages.get("is_selected") or []
        passage = next((texts[i] for i, sel in enumerate(selected) if sel == 1), None)
        if not passage:
            passage = texts[0] if texts else None
        if not passage:
            continue
        out.append(
            {
                "instruction": query,
                "search_query": query,
                "search_results": passage.strip()[:800],
                "response": answer,
            }
        )
    import random

    random.seed(42)
    random.shuffle(out)
    out = out[:18000]  # cap so it doesn't dominate the mix -- the goal is
    # to teach the search PATTERN, not to make every example a search example

    path = os.path.join(BASE, "ms_marco_search_pairs.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"ms_marco search pairs: {len(out)} -> {path}")


if __name__ == "__main__":
    convert()
