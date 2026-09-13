#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
audit_dir="test/output/collision_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$audit_dir"
for alignment in align_rows no-align_rows; do
  RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 python test/benchmark_perception_terrain.py \
    --headless --num_envs 4096 --mesh boxes --stack_power 29 \
    --"$alignment" --collision_audit --warmup 48 --steps 24 \
    --checkpoint logs/rsl_rl/rough_12dof_perception_rnn/2026-09-08_13-44-33_Perf-PureBoxes/model_199.pt \
    --output "$audit_dir/$alignment.json"
  test -s "$audit_dir/$alignment.json"
done
echo "Results: $audit_dir"
