"""
Generate synthetic training data for Kernel AI's from-scratch model using
a free, strong open-weight model (gpt-oss-120b via Cloudflare Workers AI)
PURELY as a data-generation tool — this model never answers end users,
it only ever produces examples that we then train our own model on.

Three categories, batched (multiple examples per call, asked for as JSON,
for efficiency) and saved incrementally so progress survives interruption:
  - general: diverse short instruction/answer pairs (everyday topics)
  - coding: simple runnable HTML/JS single-file snippets, artifact-style
  - kernel: Kernel-language task->code pairs AND code->explanation pairs,
    generated with the real grammar spec so the syntax is actually valid
"""

import json
import os
import sys
import time
import httpx

sys.path.insert(0, os.path.dirname(__file__))
from kernel_spec import KERNEL_SPEC

WORKER_URL = "https://kernel-ai-datagen.pabilando-karenpv.workers.dev/"
OUT_DIR = os.path.join(os.path.dirname(__file__), "../data/distilled")
os.makedirs(OUT_DIR, exist_ok=True)


def call_teacher(prompt, system, retries=3):
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=60.0) as client:
                r = client.post(WORKER_URL, json={"prompt": prompt, "system": system})
                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                return content
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def extract_json_array(text):
    """Teacher models often wrap JSON in prose or markdown code fences; find
    the array and tolerate minor formatting noise around it."""
    if text is None:
        return None
    # strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else text[0]
        text = text.lstrip("json").lstrip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        print(f"  json error: {e}; raw tail: {text[max(0, end-200):end+1]!r}")
        return None


def append_jsonl(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


GENERAL_SYSTEM = (
    "You generate training data for a small language model. Output ONLY a "
    "JSON array, no prose before or after. Each item: "
    '{"instruction": "...", "response": "..."}. Keep responses under 80 '
    "words, clear and correct."
)

CODING_SYSTEM = (
    "You generate training data for a small coding assistant. Output ONLY "
    "a JSON array, no prose before or after. Each item: "
    '{"instruction": "...", "response": "..."}. The response should contain '
    "a short, SIMPLE, fully runnable single-file HTML snippet (inline CSS/JS, "
    "under 40 lines) wrapped in a fenced code block tagged ```artifact-html. "
    "Keep it genuinely simple (a counter, a color picker, a basic to-do list, "
    "a simple animation) — not an ambitious full game."
)

KERNEL_TASK_SYSTEM = (
    "You generate training data for a small model learning a custom "
    "programming language called Kernel. Output ONLY a JSON array, no prose "
    "before or after. Each item: "
    '{"instruction": "Write a Kernel program that ...", "response": "<kernel code>"}. '
    "The response must be ONLY valid Kernel code following this exact spec, "
    "nothing else:\n" + KERNEL_SPEC
)

KERNEL_EXPLAIN_SYSTEM = (
    "You generate training data for a small model learning a custom "
    "programming language called Kernel. Output ONLY a JSON array, no prose "
    "before or after. Each item: "
    '{"instruction": "Explain what this Kernel code does:\\n<code>", "response": "<plain English explanation>"}. '
    "First invent a short valid Kernel program following this exact spec, then "
    "explain it in plain English:\n" + KERNEL_SPEC
)

TOPICS_GENERAL = [
    "everyday science facts", "basic history questions", "cooking tips",
    "simple how-to advice", "word definitions", "basic math concepts",
    "health and fitness basics", "travel tips", "productivity advice",
    "common misconceptions", "nature and animals", "basic finance questions",
]

TOPICS_CODING = [
    "a simple counter app", "a color picker", "a basic to-do list",
    "a simple timer/stopwatch", "a random quote generator", "a basic calculator",
    "a simple image gallery toggle", "a dark mode switch", "a simple form validator",
    "a basic drawing canvas", "a simple accordion/FAQ widget", "a progress bar",
]


def generate_batch(category, system, topic_hint, n=5):
    prompt = f"Generate {n} diverse JSON training examples. Topic/style hint: {topic_hint}."
    raw = call_teacher(prompt, system)
    if raw is None:
        return []
    examples = extract_json_array(raw)
    if examples is None:
        print(f"  [warn] could not parse JSON for topic '{topic_hint}', skipping")
        return []
    return examples


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["general", "coding", "kernel_task", "kernel_explain"], required=True)
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    if args.category == "general":
        out_path = os.path.join(OUT_DIR, "general.jsonl")
        system = GENERAL_SYSTEM
        topics = TOPICS_GENERAL
    elif args.category == "coding":
        out_path = os.path.join(OUT_DIR, "coding.jsonl")
        system = CODING_SYSTEM
        topics = TOPICS_CODING
    elif args.category == "kernel_task":
        out_path = os.path.join(OUT_DIR, "kernel_task.jsonl")
        system = KERNEL_TASK_SYSTEM
        topics = ["loops and counters", "simple arithmetic functions", "conditional logic",
                  "while loops with counters", "nested loops", "function calls"]
    else:
        out_path = os.path.join(OUT_DIR, "kernel_explain.jsonl")
        system = KERNEL_EXPLAIN_SYSTEM
        topics = ["loops and counters", "simple arithmetic functions", "conditional logic",
                  "while loops with counters", "nested loops", "function calls"]

    total = 0
    for i in range(args.rounds):
        topic = topics[i % len(topics)]
        print(f"round {i+1}/{args.rounds} ({args.category}, topic: {topic})...")
        examples = generate_batch(args.category, system, topic, n=8)
        if examples:
            append_jsonl(out_path, examples)
            total += len(examples)
            print(f"  got {len(examples)} examples, total so far: {total}")
        time.sleep(1)

    print(f"\nDONE. {total} examples written to {out_path}")


if __name__ == "__main__":
    main()
