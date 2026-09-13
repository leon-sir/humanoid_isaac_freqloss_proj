#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trace_dir="test/output/gpu_trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$trace_dir"
for alignment in align_rows no-align_rows; do
  RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "$trace_dir/$alignment" \
    python test/benchmark_perception_terrain.py \
      --headless --num_envs 4096 --mesh boxes --stack_power 29 \
      --"$alignment" --gpu_trace --warmup 48 --steps 24 \
      --checkpoint logs/rsl_rl/rough_12dof_perception_rnn/2026-09-08_13-44-33_Perf-PureBoxes/model_199.pt \
      --output "$trace_dir/$alignment.json"
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
    --format csv --output "$trace_dir/${alignment}_summary" \
    "$trace_dir/$alignment.nsys-rep"
done
echo "Results: $trace_dir"
