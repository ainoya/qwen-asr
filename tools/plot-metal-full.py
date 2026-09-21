#!/usr/bin/env python3
"""Plot the controlled native Metal comparison (requires matplotlib).

Usage: python3 tools/plot-metal-full.py benchmarks/metal-full-m1-pro-2026-09-21.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(sys.argv[1])
data = json.loads(path.read_text())
matplotlib.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
modes = [("0", "CPU", "#64748b"), ("1", "Hybrid Metal", "#3b82f6"),
         ("decode", "Resident generation", "#8b5cf6"), ("full", "Full Metal", "#059669")]
for ax, clip in zip(axes.flat, data["clips"]):
    values = [clip["medians"][mode]["total_ms"] / 1000 for mode, _, _ in modes]
    bars = ax.barh([label for _, label, _ in modes], values,
                   color=[color for _, _, color in modes], height=0.65)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:.3f} s" for v in values], padding=5)
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_xlabel("Inference time (seconds; lower is better)")
    gain = (1 - values[-1] / values[0]) * 100
    ax.set_title(f"{clip['label']}\nFull vs CPU: {gain:.1f}% less time", loc="left", fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.18)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle("Native Qwen3-ASR 1.7B · Apple M1 Pro\nThree measured rounds per mode, after warmup", fontsize=14)
target = path.with_suffix(".svg")
fig.savefig(target, metadata={"Date": None})
fig.savefig(target.with_suffix(".png"), dpi=150)
print(target)
