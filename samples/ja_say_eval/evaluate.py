#!/usr/bin/env python3
"""Evaluate qwen_asr on the deterministic macOS Say Japanese set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import unicodedata
from collections import defaultdict
from pathlib import Path


TIMING_PATTERNS = {
    "mel_ms": re.compile(r"^\s*Mel: .*\(([0-9.]+) ms\)$", re.MULTILINE),
    "encoder_ms": re.compile(r"^\s*Encoder: \d+ tokens \(([0-9.]+) ms\)$", re.MULTILINE),
    "prefill_ms": re.compile(r"^\s*Prefill: \d+ tokens \(([0-9.]+) ms\)$", re.MULTILINE),
    "decode_ms": re.compile(r"^\s*Decode: \d+ tokens \(([0-9.]+) ms,", re.MULTILINE),
    "ms_per_token": re.compile(r"^\s*Decode: \d+ tokens \([0-9.]+ ms, ([0-9.]+) ms/token\)$", re.MULTILINE),
    "inference_ms": re.compile(r"^Inference: ([0-9.]+) ms,", re.MULTILINE),
}


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    chars = [ch if ch.isalnum() or ch.isspace() else " " for ch in text]
    return " ".join("".join(chars).split())


def timing(stderr: str, name: str) -> float | None:
    match = TIMING_PATTERNS[name].search(stderr)
    return float(match.group(1)) if match else None


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="../../qwen_asr")
    parser.add_argument("--model-dir", default="../../qwen3-asr-1.7b")
    parser.add_argument("--weights", choices=("q8", "q8-lm", "bf16"), default="q8-lm")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--profile", help="Only run one profile, such as kyoko_normal")
    parser.add_argument("--baseline", help="JSON result whose predictions are the differential reference")
    parser.add_argument("--output", help="Write the complete JSON result")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    def resolve_input(value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        from_cwd = (Path.cwd() / path).resolve()
        return from_cwd if from_cwd.exists() else (here / path).resolve()

    binary = resolve_input(args.binary)
    model_dir = resolve_input(args.model_dir)
    rows = list(csv.DictReader((here / "manifest.tsv").open(encoding="utf-8"), delimiter="\t"))
    if args.profile:
        rows = [row for row in rows if row["profile"] == args.profile]
    if not rows:
        raise SystemExit("no matching samples")

    baseline_predictions: dict[str, str] = {}
    if args.baseline:
        baseline_path = resolve_input(args.baseline)
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_predictions = {
            item["sample_id"]: item["prediction"] for item in baseline_payload["results"]
        }

    results = []
    for index, row in enumerate(rows, 1):
        wav = here / row["wav"]
        reference = (here / row["reference"]).read_text(encoding="utf-8").strip()
        cmd = [
            str(binary), "-d", str(model_dir), "-i", str(wav),
            "--silent", "--debug", "--language", "Japanese",
            "--weights", args.weights, "-t", str(args.threads), "-S", "0",
        ]
        started = time.monotonic()
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout, check=False)
        wall_ms = (time.monotonic() - started) * 1000
        if proc.returncode != 0:
            print(proc.stderr)
            raise SystemExit(f"{row['sample_id']}: inference failed with {proc.returncode}")

        prediction = proc.stdout.strip()
        ref_norm, pred_norm = normalize(reference), normalize(prediction)
        distance = levenshtein(pred_norm, ref_norm)
        result = {
            **row,
            "reference_text": reference,
            "prediction": prediction,
            "normalized_distance": distance,
            "normalized_chars": max(1, len(ref_norm)),
            "normalized_cer": distance / max(1, len(ref_norm)),
            "wall_ms": wall_ms,
            **{name: timing(proc.stderr, name) for name in TIMING_PATTERNS},
        }
        if row["sample_id"] in baseline_predictions:
            base_norm = normalize(baseline_predictions[row["sample_id"]])
            result["baseline_distance"] = levenshtein(pred_norm, base_norm)
            result["baseline_chars"] = max(1, len(base_norm))
            result["baseline_cer"] = result["baseline_distance"] / result["baseline_chars"]
            result["baseline_exact"] = prediction == baseline_predictions[row["sample_id"]]
        results.append(result)
        print(
            f"[{index:02d}/{len(rows):02d}] {row['sample_id']:<30} "
            f"CER={result['normalized_cer']:.3f} "
            f"enc={result['encoder_ms'] or 0:.0f}ms "
            f"prefill={result['prefill_ms'] or 0:.0f}ms "
            f"decode={result['decode_ms'] or 0:.0f}ms"
        )

    total_dist = sum(item["normalized_distance"] for item in results)
    total_chars = sum(item["normalized_chars"] for item in results)
    stage_names = ("mel_ms", "encoder_ms", "prefill_ms", "decode_ms", "inference_ms")
    summary = {
        "samples": len(results),
        "weights": args.weights,
        "threads": args.threads,
        "micro_normalized_cer": total_dist / max(1, total_chars),
        "macro_normalized_cer": sum(item["normalized_cer"] for item in results) / len(results),
        **{name: sum(item[name] or 0 for item in results) for name in stage_names},
    }
    baseline_items = [item for item in results if "baseline_distance" in item]
    if baseline_items:
        baseline_dist = sum(item["baseline_distance"] for item in baseline_items)
        baseline_chars = sum(item["baseline_chars"] for item in baseline_items)
        summary["micro_baseline_cer"] = baseline_dist / max(1, baseline_chars)
        summary["baseline_exact_match_rate"] = (
            sum(bool(item["baseline_exact"]) for item in baseline_items) / len(baseline_items)
        )
    by_category: dict[str, dict[str, float | int]] = {}
    grouped = defaultdict(list)
    for item in results:
        grouped[item["category"]].append(item)
    for category, items in sorted(grouped.items()):
        dist = sum(item["normalized_distance"] for item in items)
        chars = sum(item["normalized_chars"] for item in items)
        by_category[category] = {"samples": len(items), "micro_normalized_cer": dist / max(1, chars)}

    payload = {"summary": summary, "by_category": by_category, "results": results}
    print("\nsummary")
    print(json.dumps({"summary": summary, "by_category": by_category}, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output).resolve()
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
