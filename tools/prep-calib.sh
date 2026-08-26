#!/bin/bash
# Build a calibration set for weight quantization from a directory of recordings.
#
#   tools/prep-calib.sh <src-dir> <out-dir> [minutes] [seconds-per-clip]
#
# Quantization calibration needs audio, not transcripts: it only collects
# per-channel activation magnitudes from forward passes. So any recordings of
# the kind the engine will actually see will do, and no labelling is required.
#
# Variety matters more than volume - a few hundred distinct utterances pin the
# statistics down better than one long recording - so clips are trimmed and
# spread across the source's duration range.
#
# The output is derived from whatever you point it at. If that is private,
# point it somewhere outside the repository.
set -e
SRC=${1:?usage: prep-calib.sh <src-dir> <out-dir> [minutes] [sec-per-clip]}
OUT=${2:?usage: prep-calib.sh <src-dir> <out-dir> [minutes] [sec-per-clip]}
BUDGET_MIN=${3:-30}
CLIP_SEC=${4:-45}

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
mkdir -p "$OUT"

# Sort by duration so the pick spans short and long material rather than
# whatever the filesystem happens to return first.
listing=$(mktemp)
find "$SRC" -type f \( -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.aac' \
     -o -iname '*.wav' -o -iname '*.caf' -o -iname '*.flac' \) -print0 |
while IFS= read -r -d '' f; do
    # A single unreadable or slow-to-materialise input must not stall the run.
    d=$(perl -e 'alarm 10; exec @ARGV' ffprobe -v error \
            -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null || echo 0)
    printf '%s\t%s\n' "${d:-0}" "$f"
done | sort -n > "$listing"

total=$(wc -l < "$listing")
[ "$total" -gt 0 ] || { echo "no audio found under $SRC" >&2; exit 1; }

budget=$((BUDGET_MIN * 60))
want=$(( budget / CLIP_SEC ))
[ "$want" -lt 1 ] && want=1
step=$(( total / want )); [ "$step" -lt 1 ] && step=1

n=0; secs=0; i=0
while IFS=$'\t' read -r dur path; do
    i=$((i + 1))
    [ $(( (i - 1) % step )) -eq 0 ] || continue
    # Skip anything too short to carry useful statistics.
    awk "BEGIN{exit !($dur < 3)}" && continue
    out=$(printf '%s/calib_%04d.wav' "$OUT" "$n")
    perl -e 'alarm 120; exec @ARGV' ffmpeg -v error -y -i "$path" -t "$CLIP_SEC" \
            -ac 1 -ar 16000 -c:a pcm_s16le "$out" 2>/dev/null || { rm -f "$out"; continue; }
    # 16 kHz mono s16le: two bytes per sample, 44-byte header.
    bytes=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out")
    secs=$(awk "BEGIN{print $secs + ($bytes - 44) / 32000}")
    n=$((n + 1))
    awk "BEGIN{exit !($secs >= $budget)}" && break
done < "$listing"
rm -f "$listing"

echo "wrote $n clips, $(awk "BEGIN{printf \"%.1f\", $secs/60}") min of 16 kHz mono to $OUT"
