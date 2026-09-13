#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trace_dir="test/output/offset_gpu_trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$trace_dir"
for variant in automatic contact_only rest_only both; do
  offset_args=()
  case "$variant" in
    contact_only) offset_args=(--terrain_contact_offset 0.02) ;;
    rest_only) offset_args=(--terrain_rest_offset 0) ;;
    both) offset_args=(--terrain_contact_offset 0.02 --terrain_rest_offset 0) ;;
  esac
  RSL_RL_DREAM_ENABLE_CUDNN_RNN=1 nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "$trace_dir/$variant" \
    python test/benchmark_perception_terrain.py \
      --headless --num_envs 4096 --mesh boxes --stack_power 29 --align_rows \
      "${offset_args[@]}" --collision_audit --gpu_trace --warmup 48 --steps 24 \
      --checkpoint logs/rsl_rl/rough_12dof_perception_rnn/2026-09-08_13-44-33_Perf-PureBoxes/model_199.pt \
      --output "$trace_dir/$variant.json"
  test -s "$trace_dir/$variant.json"
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
    --format csv --output "$trace_dir/${variant}_summary" \
    "$trace_dir/$variant.nsys-rep"
done
echo "Results: $trace_dir"
