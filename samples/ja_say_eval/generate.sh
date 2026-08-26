#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h}
cases_file="$root_dir/cases.tsv"
audio_dir="$root_dir/audio"
manifest="$root_dir/manifest.tsv"

if [[ ! -x /usr/bin/say ]]; then
  print -u2 "error: /usr/bin/say is required"
  exit 1
fi
if [[ ! -x /usr/bin/afconvert ]]; then
  print -u2 "error: /usr/bin/afconvert is required"
  exit 1
fi
if [[ ! -f "$cases_file" ]]; then
  print -u2 "error: missing $cases_file"
  exit 1
fi

mkdir -p "$audio_dir"

# profile_id|voice|rate. These voices ship with current macOS Japanese voices.
# Fail clearly instead of silently substituting another voice, so regeneration
# remains comparable across machines.
profiles=(
  'kyoko_normal|Kyoko|180'
  'eddy_fast|Eddy (日本語（日本）)|215'
  'grandpa_slow|Grandpa (日本語（日本）)|155'
)

available_voices=$(/usr/bin/say -v '?')
for profile in $profiles; do
  IFS='|' read -r profile_id voice rate <<< "$profile"
  if ! print -r -- "$available_voices" | grep -Fq -- "$voice"; then
    print -u2 "error: required voice is not installed: $voice"
    exit 1
  fi
done

print -r -- $'sample_id\tcase_id\tcategory\tprofile\tvoice\trate\twav\treference' > "$manifest"

tail -n +2 "$cases_file" | while IFS=$'\t' read -r case_id category text; do
  [[ -n "$case_id" ]] || continue
  for profile in $profiles; do
    IFS='|' read -r profile_id voice rate <<< "$profile"
    sample_id="${case_id}__${profile_id}"
    aiff="$audio_dir/${sample_id}.aiff"
    wav="$audio_dir/${sample_id}.wav"
    ref="$audio_dir/${sample_id}.txt"

    print "generating $sample_id"
    /usr/bin/say -v "$voice" -r "$rate" -o "$aiff" "$text"
    /usr/bin/afconvert -f WAVE -d LEI16@16000 "$aiff" "$wav"
    rm "$aiff"
    print -r -- "$text" > "$ref"

    print -r -- "$sample_id"$'\t'"$case_id"$'\t'"$category"$'\t'"$profile_id"$'\t'"$voice"$'\t'"$rate"$'\t'"audio/${sample_id}.wav"$'\t'"audio/${sample_id}.txt" >> "$manifest"
  done
done

print "wrote $manifest"
