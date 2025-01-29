#!/bin/bash
set -e

##############################################################################
# Usage:
#   ./train.sh [--mode simulation|emulation]
#
# Description:
#   - If --mode is "simulation" (or omitted), runs the simulation-based
#     flow using 'src/simulator/abr_simulator/pensieve/genet.py'.
#   - If --mode is "emulation", runs the provided emulation command.
##############################################################################

# Default mode
MODE="simulation"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift
      ;;
    *)  # unknown option
      echo "Unknown option: $1"
      echo "Usage: $0 [--mode simulation|emulation]"
      exit 1
      ;;
  esac
  shift
done

# Common variables
save_dir=results/abr
video_size_file_dir=data/abr/video_sizes
val_trace_dir=data/abr/val_FCC
config_file=config/abr/udr3.json
pretrained_model=results/abr/new_trace_gen/udr3/seed_10/model_saved/nn_model_ep_1000.ckpt

if [ "$MODE" = "simulation" ]; then
  echo "Running in simulation mode..."
  for seed in 10 20 30; do
      mkdir -p "${save_dir}/genet_mpc/seed_${seed}"
      python src/simulator/abr_simulator/pensieve/genet.py \
          --save-dir "${save_dir}/genet_mpc/seed_${seed}" \
          --heuristic mpc \
          --seed "${seed}" \
          --video-size-file-dir "${video_size_file_dir}" \
          --config-file "${config_file}" \
          --model-path "${pretrained_model}" \
          --val-trace-dir "${val_trace_dir}"
  done

elif [ "$MODE" = "emulation" ]; then
  echo "Running in emulation mode..."
  # Run your specific emulation command here:
  python src/emulator/abr/pensieve/agent_policy/train.py \
      --total-epoch=100 \
      --seed=10 \
      --save-dir=results/abr/genet_mpc/seed_10/pensieve_train \
      --exp-name=pensieve_train \
      --model-path=results/abr/new_trace_gen/udr3/seed_10/model_saved/nn_model_ep_1000.ckpt \
      --nagent=1 \
      --video-size-file-dir=/users/janechen/Genet/data/abr/video_sizes \
      --val-freq=100 \
      --train-trace-dir=/users/janechen/Genet/data/abr/trace_set_1
      # udr \
      # --config-file=config/abr/udr3.json \
      # --val-trace-dir=data/abr/val_FCC

else
  echo "Unknown mode: $MODE"
  echo "Usage: $0 [--mode simulation|emulation]"
  exit 1
fi
