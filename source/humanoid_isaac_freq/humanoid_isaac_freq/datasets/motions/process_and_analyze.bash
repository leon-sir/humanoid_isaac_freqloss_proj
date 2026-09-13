#!/usr/bin/env bash
# Use the active Isaac Lab conda environment. Regenerate derived CSV/JSON outputs.
set -euo pipefail

data_name="${1:-Neutral_walk_forward_002__A057}"
if [[ "$data_name" == "--help" || "$data_name" == "-h" ]]; then
  echo "Usage: bash process_and_analyze.bash [Neutral_walk_forward_002__A057] [collector options...]"
  echo "Example extra options: --headless --fast"
  exit 0
fi
if (( $# > 0 )); then shift; fi
if [[ ! "$data_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Invalid motion name: use the directory name only, without .csv or slashes." >&2
  exit 1
fi

motion_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$motion_root/../../../../.." && pwd)"
cd -- "$project_root"
motion_dir="$motion_root/$data_name"
raw_csv="$motion_dir/$data_name.csv"
periodic_csv="$motion_dir/${data_name}_periodic.csv"
filtered_csv="$motion_dir/${data_name}_periodic_filtered.csv"
[[ -f "$raw_csv" ]] || { echo "Missing input: $raw_csv" >&2; exit 1; }

echo "[1/4] Prepare periodic motion: $data_name"
python scripts/collector/process_mocap_data.py \
  --csv "$raw_csv" --fps 120 --min_period 0.7 --max_period 1.6 \
  --trim 0.5 --duration 30 --output "$periodic_csv" --overwrite

# Restrict spectrum selection to a new, completed export from THIS step.
marker="$(mktemp)"
trap 'rm -f -- "$marker"' EXIT
echo "[2/4] Replay periodic motion, record video and analyze spectrum"
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv "$periodic_csv" --source_fps 120 --sample_fps 50 \
  --warmup_steps 250 --num_frames 750 --max_frequency_hz 5 \
  --video --fast_exit "$@"

spectrum_csv="$(python - "$periodic_csv" "$marker" <<'PY'
import json
import sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
started = Path(sys.argv[2]).stat().st_mtime_ns
matches = []
for metadata in Path('scripts/output/spectrum_mocap_analysis').glob('*/rollout_metadata.json'):
    if metadata.stat().st_mtime_ns < started:
        continue
    payload = json.loads(metadata.read_text())
    spectrum = metadata.parent / 'runner_joint_power_scale.csv'
    if Path(payload['csv']).resolve() == source and spectrum.is_file():
        matches.append(spectrum.resolve())
if len(matches) != 1:
    raise SystemExit(f'Expected exactly one new completed export for {source}; found {len(matches)}. Do not run this motion concurrently.')
print(matches[0])
PY
)"
echo "Spectrum: $spectrum_csv"

echo "[3/4] Keep DC + gait fundamental"
python scripts/collector/filter_mocap_fundamental.py \
  --csv "$periodic_csv" --spectrum "$spectrum_csv" --fps 120 --output "$filtered_csv" \
  --symmetry --overwrite

echo "[4/4] Replay filtered motion, record video and analyze spectrum"
python scripts/collector/collect_mocap_data_analysis_scale.py \
  --csv "$filtered_csv" --source_fps 120 --sample_fps 50 \
  --warmup_steps 250 --num_frames 750 --max_frequency_hz 5 \
  --video --fast_exit "$@"
echo "Done. Filtered motion: $filtered_csv"
