"""
Kernel (.kn) corpus builder.

Generates N random programs, runs each through the real `kernel trace`
binary (the ground-truth oracle, same interpreter the language ships),
and keeps only the ones that run successfully. Programs that fail
(checker rejection, runtime panic, e.g. integer overflow in debug mode)
are counted and reported honestly, not silently dropped.

Usage:
    python3 build_corpus.py --n 500 --out ../data/corpus.jsonl \
        --kernel-bin ~/projects/kernel-lang/target/debug/kernel
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os

sys.path.insert(0, os.path.dirname(__file__))
from generate_programs import ProgramGenerator


def run_one(kernel_bin, source):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".kn", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        result = subprocess.run(
            [kernel_bin, "trace", path],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        try:
            data = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            return None, f"bad JSON: {e}: {result.stdout[:200]}"
        return data, None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        os.unlink(path)


def build(n, out_path, kernel_bin, seed_start=0, max_depth=3, max_stmts=10):
    successes = 0
    failures = 0
    failure_reasons = {}

    with open(out_path, "w") as out:
        for i in range(n):
            gen = ProgramGenerator(seed=seed_start + i)
            source = gen.gen_program(max_depth=max_depth, max_stmts=max_stmts)
            data, err = run_one(kernel_bin, source)
            if data is None:
                failures += 1
                key = err.split("\n")[0][:80] if err else "unknown"
                failure_reasons[key] = failure_reasons.get(key, 0) + 1
                continue
            record = {
                "id": seed_start + i,
                "source": source,
                "output": data["output"],
                "trace": data["trace"],
                "final_value": data["final_value"],
            }
            out.write(json.dumps(record) + "\n")
            successes += 1

    print(f"built {out_path}")
    print(f"  successes: {successes}")
    print(f"  failures:  {failures}  ({failures / n * 100:.1f}%)")
    if failure_reasons:
        print("  failure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:5d}  {reason}")
    return successes, failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="../data/corpus.jsonl")
    ap.add_argument("--kernel-bin", required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-stmts", type=int, default=10)
    args = ap.parse_args()
    build(args.n, args.out, args.kernel_bin, args.seed_start, args.max_depth, args.max_stmts)
