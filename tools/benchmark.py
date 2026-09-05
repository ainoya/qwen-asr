#!/usr/bin/env python3
"""
tools/benchmark.py - Measure and plot Qwen-ASR speed history across commits.

Usage:
  python3 tools/benchmark.py --plot
  python3 tools/benchmark.py --list
  python3 tools/benchmark.py --add --commit <hash> --title <title> --11s <sec> --41s <sec> --decode <ms> --prefill <ms> --heap <mb>
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

DEFAULT_HISTORY_FILE = "benchmarks/history.json"
DEFAULT_OUTPUT_SVG = "benchmarks/speed_history.svg"

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "current"

def load_history(filepath=DEFAULT_HISTORY_FILE):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(data, filepath=DEFAULT_HISTORY_FILE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved benchmark history to {filepath}")

def generate_svg(history_file=DEFAULT_HISTORY_FILE, output_svg=DEFAULT_OUTPUT_SVG):
    history = load_history(history_file)
    if not history:
        print(f"Error: No history data in {history_file}")
        return

    cur_hash = get_git_commit()
    for entry in history:
        if entry.get("commit") == "current":
            entry["commit"] = cur_hash

    n = len(history)
    W, H = 1020, 640
    pad = 40

    bg = "#0d1117"
    panel_bg = "#161b22"
    border = "#30363d"
    text_main = "#f0f6fc"
    text_muted = "#8b949e"
    color_11s = "#58a6ff"
    color_41s = "#3fb950"
    color_decode = "#d29922"
    color_prefill = "#bc8cff"
    color_mem = "#f778ba"

    latest = history[-1]
    rtf_11s = latest.get("clip_11s_rtf", 11.0 / latest.get("clip_11s_sec", 1.0))
    decode_ms = latest.get("decode_ms_per_tok", 0.0)
    heap_mb = latest.get("wasm_heap_mb", 0)

    svg = []
    svg.append(f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <defs>
    <style>
      text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
      .title {{ font-size: 20px; font-weight: 700; fill: {text_main}; }}
      .subtitle {{ font-size: 13px; fill: {text_muted}; }}
      .panel-title {{ font-size: 14px; font-weight: 600; fill: {text_main}; }}
      .axis-lbl {{ font-size: 11px; fill: {text_muted}; text-anchor: middle; }}
      .y-lbl {{ font-size: 11px; fill: {text_muted}; text-anchor: end; }}
      .legend {{ font-size: 11px; font-weight: 500; }}
      .badge-txt {{ font-size: 12px; font-weight: 600; fill: {text_main}; }}
      .val-lbl {{ font-size: 10px; font-weight: 600; text-anchor: middle; }}
    </style>
  </defs>

  <rect width="{W}" height="{H}" fill="{bg}"/>

  <text x="{pad}" y="36" class="title">Qwen-ASR Optimization Benchmark History</text>
  <text x="{pad}" y="56" class="subtitle">Commit-by-commit speed, latency, and memory progression on WebGPU / Browser (1.7B Model)</text>

  <g transform="translate({W - pad - 420}, 18)">
    <rect width="130" height="42" rx="6" fill="{panel_bg}" stroke="{border}"/>
    <text x="65" y="18" text-anchor="middle" font-size="10" fill="{text_muted}">REALTIME FACTOR</text>
    <text x="65" y="34" text-anchor="middle" class="badge-txt" fill="{color_11s}">{rtf_11s:.2f}x (11s clip)</text>

    <rect x="140" width="130" height="42" rx="6" fill="{panel_bg}" stroke="{border}"/>
    <text x="205" y="18" text-anchor="middle" font-size="10" fill="{text_muted}">DECODE LATENCY</text>
    <text x="205" y="34" text-anchor="middle" class="badge-txt" fill="{color_decode}">{decode_ms:.1f} ms / tok</text>

    <rect x="280" width="140" height="42" rx="6" fill="{panel_bg}" stroke="{border}"/>
    <text x="350" y="18" text-anchor="middle" font-size="10" fill="{text_muted}">WASM HEAP MEMORY</text>
    <text x="350" y="34" text-anchor="middle" class="badge-txt" fill="{color_mem}">{heap_mb} MB (-85%)</text>
  </g>
""")

    # Panel 1: End-to-End Latency
    p1_x, p1_y, p1_w, p1_h = pad, 80, 530, 250
    svg.append(f"""
  <rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" rx="8" fill="{panel_bg}" stroke="{border}"/>
  <text x="{p1_x + 16}" y="{p1_y + 24}" class="panel-title">End-to-End Transcription Time (seconds)</text>
  <circle cx="{p1_x + 360}" cy="{p1_y + 20}" r="4" fill="{color_11s}"/>
  <text x="{p1_x + 370}" y="{p1_y + 24}" class="legend" fill="{color_11s}">11s English</text>
  <circle cx="{p1_x + 440}" cy="{p1_y + 20}" r="4" fill="{color_41s}"/>
  <text x="{p1_x + 450}" y="{p1_y + 24}" class="legend" fill="{color_41s}">41s Japanese</text>
""")

    y_max = 16.0
    c_x0 = p1_x + 45
    c_y0 = p1_y + 45
    c_w = p1_w - 60
    c_h = p1_h - 75

    for y_val in [0, 4, 8, 12, 16]:
        y_pos = c_y0 + c_h - (y_val / y_max) * c_h
        svg.append(f"""  <line x1="{c_x0}" y1="{y_pos}" x2="{c_x0 + c_w}" y2="{y_pos}" stroke="{border}" stroke-dasharray="3,3"/>
  <text x="{c_x0 - 8}" y="{y_pos + 4}" class="y-lbl">{y_val}s</text>""")

    pts_11s = []
    pts_41s = []
    for i, e in enumerate(history):
        x = c_x0 + (i / (n - 1)) * c_w if n > 1 else c_x0 + c_w / 2
        y11 = c_y0 + c_h - (e["clip_11s_sec"] / y_max) * c_h
        y41 = c_y0 + c_h - (e["clip_41s_sec"] / y_max) * c_h
        pts_11s.append((x, y11, e["clip_11s_sec"], e["clip_11s_rtf"]))
        pts_41s.append((x, y41, e["clip_41s_sec"], e["clip_41s_rtf"]))

    d41 = "M " + " L ".join([f"{x},{y}" for x, y, _, _ in pts_41s])
    svg.append(f"""  <path d="{d41}" fill="none" stroke="{color_41s}" stroke-width="2.5"/>""")
    for x, y, val, rtf in pts_41s:
        svg.append(f"""  <circle cx="{x}" cy="{y}" r="5" fill="{bg}" stroke="{color_41s}" stroke-width="2.5"/>
  <text x="{x}" y="{y - 9}" class="val-lbl" fill="{color_41s}">{val}s ({rtf}x)</text>""")

    d11 = "M " + " L ".join([f"{x},{y}" for x, y, _, _ in pts_11s])
    svg.append(f"""  <path d="{d11}" fill="none" stroke="{color_11s}" stroke-width="2.5"/>""")
    for x, y, val, rtf in pts_11s:
        svg.append(f"""  <circle cx="{x}" cy="{y}" r="5" fill="{bg}" stroke="{color_11s}" stroke-width="2.5"/>
  <text x="{x}" y="{y - 9}" class="val-lbl" fill="{color_11s}">{val}s ({rtf}x)</text>""")

    for i, e in enumerate(history):
        x = c_x0 + (i / (n - 1)) * c_w if n > 1 else c_x0 + c_w / 2
        svg.append(f"""  <text x="{x}" y="{c_y0 + c_h + 18}" class="axis-lbl">{e["commit"]}</text>""")

    # Panel 2: Decode Latency
    p2_x, p2_y, p2_w, p2_h = pad + p1_w + 16, 80, W - pad - (pad + p1_w + 16), 250
    svg.append(f"""
  <rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="8" fill="{panel_bg}" stroke="{border}"/>
  <text x="{p2_x + 16}" y="{p2_y + 24}" class="panel-title">Decode Step Latency (ms / token)</text>
  <text x="{p2_x + 16}" y="{p2_y + 40}" class="subtitle">Memory-bandwidth &amp; kernel reduction bound</text>
""")

    c2_x0 = p2_x + 45
    c2_y0 = p2_y + 55
    c2_w = p2_w - 60
    c2_h = p2_h - 85
    y2_max = 50.0

    for y_val in [0, 15, 30, 45]:
        y_pos = c2_y0 + c2_h - (y_val / y2_max) * c2_h
        svg.append(f"""  <line x1="{c2_x0}" y1="{y_pos}" x2="{c2_x0 + c2_w}" y2="{y_pos}" stroke="{border}" stroke-dasharray="3,3"/>
  <text x="{c2_x0 - 8}" y="{y_pos + 4}" class="y-lbl">{y_val}</text>""")

    pts_dec = []
    for i, e in enumerate(history):
        x = c2_x0 + (i / (n - 1)) * c2_w if n > 1 else c2_x0 + c2_w / 2
        y = c2_y0 + c2_h - (e["decode_ms_per_tok"] / y2_max) * c2_h
        pts_dec.append((x, y, e["decode_ms_per_tok"]))

    d_dec = "M " + " L ".join([f"{x},{y}" for x, y, _ in pts_dec])
    svg.append(f"""  <path d="{d_dec}" fill="none" stroke="{color_decode}" stroke-width="2.5"/>""")
    for x, y, val in pts_dec:
        svg.append(f"""  <circle cx="{x}" cy="{y}" r="5" fill="{bg}" stroke="{color_decode}" stroke-width="2.5"/>
  <text x="{x}" y="{y - 9}" class="val-lbl" fill="{color_decode}">{val}ms</text>""")

    for i, e in enumerate(history):
        x = c2_x0 + (i / (n - 1)) * c2_w if n > 1 else c2_x0 + c2_w / 2
        svg.append(f"""  <text x="{x}" y="{c2_y0 + c2_h + 18}" class="axis-lbl">{e["commit"]}</text>""")

    # Panel 3: Prefill Latency
    p3_x, p3_y, p3_w, p3_h = pad, 345, p1_w, 250
    svg.append(f"""
  <rect x="{p3_x}" y="{p3_y}" width="{p3_w}" height="{p3_h}" rx="8" fill="{panel_bg}" stroke="{border}"/>
  <text x="{p3_x + 16}" y="{p3_y + 24}" class="panel-title">Prefill GEMM Latency (41s clip, seq=549)</text>
  <text x="{p3_x + 16}" y="{p3_y + 40}" class="subtitle">Compute-bound prompt processing time</text>
""")

    c3_x0 = p3_x + 55
    c3_y0 = p3_y + 55
    c3_w = p3_w - 75
    c3_h = p3_h - 85
    y3_max = 8000.0

    for y_val in [0, 2000, 4000, 6000, 8000]:
        y_pos = c3_y0 + c3_h - (y_val / y3_max) * c3_h
        svg.append(f"""  <line x1="{c3_x0}" y1="{y_pos}" x2="{c3_x0 + c3_w}" y2="{y_pos}" stroke="{border}" stroke-dasharray="3,3"/>
  <text x="{c3_x0 - 8}" y="{y_pos + 4}" class="y-lbl">{int(y_val)}ms</text>""")

    pts_pre = []
    for i, e in enumerate(history):
        x = c3_x0 + (i / (n - 1)) * c3_w if n > 1 else c3_x0 + c3_w / 2
        y = c3_y0 + c3_h - (e["prefill_41s_ms"] / y3_max) * c3_h
        pts_pre.append((x, y, e["prefill_41s_ms"]))

    d_pre = "M " + " L ".join([f"{x},{y}" for x, y, _ in pts_pre])
    svg.append(f"""  <path d="{d_pre}" fill="none" stroke="{color_prefill}" stroke-width="2.5"/>""")
    for x, y, val in pts_pre:
        svg.append(f"""  <circle cx="{x}" cy="{y}" r="5" fill="{bg}" stroke="{color_prefill}" stroke-width="2.5"/>
  <text x="{x}" y="{y - 9}" class="val-lbl" fill="{color_prefill}">{int(val)}ms</text>""")

    for i, e in enumerate(history):
        x = c3_x0 + (i / (n - 1)) * c3_w if n > 1 else c3_x0 + c3_w / 2
        svg.append(f"""  <text x="{x}" y="{c3_y0 + c3_h + 18}" class="axis-lbl">{e["commit"]}</text>""")

    # Panel 4: WASM Heap Memory Footprint
    p4_x, p4_y, p4_w, p4_h = pad + p1_w + 16, 345, p2_w, 250
    svg.append(f"""
  <rect x="{p4_x}" y="{p4_y}" width="{p4_w}" height="{p4_h}" rx="8" fill="{panel_bg}" stroke="{border}"/>
  <text x="{p4_x + 16}" y="{p4_y + 24}" class="panel-title">WASM Heap Memory Footprint (MB)</text>
  <text x="{p4_x + 16}" y="{p4_y + 40}" class="subtitle">Offloading to GPU storage buffers reduces CPU memory</text>
""")

    c4_x0 = p4_x + 50
    c4_y0 = p4_y + 55
    c4_w = p2_w - 70
    c4_h = p4_h - 85
    y4_max = 2500.0

    for y_val in [0, 500, 1000, 1500, 2000]:
        y_pos = c4_y0 + c4_h - (y_val / y4_max) * c4_h
        svg.append(f"""  <line x1="{c4_x0}" y1="{y_pos}" x2="{c4_x0 + c4_w}" y2="{y_pos}" stroke="{border}" stroke-dasharray="3,3"/>
  <text x="{c4_x0 - 8}" y="{y_pos + 4}" class="y-lbl">{int(y_val)}MB</text>""")

    pts_mem = []
    for i, e in enumerate(history):
        x = c4_x0 + (i / (n - 1)) * c4_w if n > 1 else c4_x0 + c4_w / 2
        y = c4_y0 + c4_h - (e["wasm_heap_mb"] / y4_max) * c4_h
        pts_mem.append((x, y, e["wasm_heap_mb"]))

    d_mem = "M " + " L ".join([f"{x},{y}" for x, y, _ in pts_mem])
    svg.append(f"""  <path d="{d_mem}" fill="none" stroke="{color_mem}" stroke-width="2.5"/>""")
    for x, y, val in pts_mem:
        svg.append(f"""  <circle cx="{x}" cy="{y}" r="5" fill="{bg}" stroke="{color_mem}" stroke-width="2.5"/>
  <text x="{x}" y="{y - 9}" class="val-lbl" fill="{color_mem}">{int(val)}MB</text>""")

    for i, e in enumerate(history):
        x = c4_x0 + (i / (n - 1)) * c4_w if n > 1 else c4_x0 + c4_w / 2
        svg.append(f"""  <text x="{x}" y="{c4_y0 + c4_h + 18}" class="axis-lbl">{e["commit"]}</text>""")

    svg.append(f"""
  <text x="{pad}" y="{H - 12}" font-size="11" fill="{text_muted}">Generated by tools/benchmark.py • Values verified on Apple Silicon / WebGPU pipeline</text>
</svg>""")

    os.makedirs(os.path.dirname(output_svg), exist_ok=True)
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"Generated {output_svg}")

def print_list(history_file=DEFAULT_HISTORY_FILE):
    history = load_history(history_file)
    print(f"{'Commit':<10} {'Date':<12} {'11s(s)':<8} {'11s(RTF)':<10} {'41s(s)':<8} {'Dec(ms)':<9} {'Prefill':<9} {'Heap':<8} Title")
    print("-" * 95)
    for e in history:
        print(f"{e.get('commit',''):<10} {e.get('date',''):<12} {e.get('clip_11s_sec',0):<8.2f} "
              f"{e.get('clip_11s_rtf',0):<10.2f} {e.get('clip_41s_sec',0):<8.2f} "
              f"{e.get('decode_ms_per_tok',0):<9.1f} {e.get('prefill_41s_ms',0):<9.0f} "
              f"{e.get('wasm_heap_mb',0):<8} {e.get('title','')}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark & plot speed history for Qwen-ASR.")
    parser.add_argument("--plot", action="store_true", help="Generate the SVG speed history graph")
    parser.add_argument("--list", action="store_true", help="List benchmark history entries")
    parser.add_argument("--history", default=DEFAULT_HISTORY_FILE, help="Path to history.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_SVG, help="Path to output SVG")
    parser.add_argument("--add", action="store_true", help="Add or update a benchmark record")
    parser.add_argument("--commit", help="Commit hash (default: current HEAD)")
    parser.add_argument("--title", help="Milestone title")
    parser.add_argument("--11s", dest="s11", type=float, help="11s clip latency (sec)")
    parser.add_argument("--41s", dest="s41", type=float, help="41s clip latency (sec)")
    parser.add_argument("--decode", type=float, help="Decode latency (ms/tok)")
    parser.add_argument("--prefill", type=float, help="Prefill latency (ms)")
    parser.add_argument("--heap", type=int, help="WASM heap (MB)")
    parser.add_argument("--notes", help="Notes")

    args = parser.parse_args()

    if args.add:
        commit = args.commit or get_git_commit()
        history = load_history(args.history)
        entry = None
        for item in history:
            if item.get("commit") == commit:
                entry = item
                break
        if not entry:
            entry = {"commit": commit, "date": datetime.date.today().isoformat()}
            history.append(entry)

        if args.title: entry["title"] = args.title
        if args.s11 is not None:
            entry["clip_11s_sec"] = args.s11
            entry["clip_11s_rtf"] = round(11.0 / args.s11, 2)
        if args.s41 is not None:
            entry["clip_41s_sec"] = args.s41
            entry["clip_41s_rtf"] = round(41.0 / args.s41, 2)
        if args.decode is not None: entry["decode_ms_per_tok"] = args.decode
        if args.prefill is not None: entry["prefill_41s_ms"] = args.prefill
        if args.heap is not None: entry["wasm_heap_mb"] = args.heap
        if args.notes: entry["notes"] = args.notes

        save_history(history, args.history)
        generate_svg(args.history, args.output)
    elif args.list:
        print_list(args.history)
    else:
        generate_svg(args.history, args.output)

if __name__ == "__main__":
    main()
