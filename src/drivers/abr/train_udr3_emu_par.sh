#!/bin/bash
##############################################################################
# Usage:
# ./train_udr3_emu_par.sh [--mode simulation|emulation]
#
# Description:
# - If --mode is "simulation" (or omitted), runs the simulation-based
# flow using 'src/simulator/abr_simulator/pensieve/train.py'.
# - If --mode is "emulation", runs the provided emulation command.
# - Use --emulation-seed to specify the seed for emulation mode (default: 10).
# - Use --adaptor-input and --adaptor-hidden-layer to specify emulation parameters.
##############################################################################
set -e
rm -rf /mydata/results/*
rm -rf /mydata/logs/*

save_dir=/mydata/results/abr
video_size_file_dir=data/abr/video_sizes
val_trace_dir=data/abr/val_FCC
total_epoch=75000
train_name=udr3_emu_par
config_file=config/abr/${train_name}.json
original_model_path=fig_reproduce/data/all_models/udr_3/nn_model_ep_58000.ckpt

# Default mode
MODE="simulation"
emulation_seed=10
adaptor_input=""
adaptor_hidden_layer=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
case $1 in
    --mode)
        MODE="$2"
        shift
        ;;
    --emulation-seed)
        emulation_seed="$2"
        shift
        ;;
    --adaptor-input)
        adaptor_input="$2"
        shift
        ;;
    --adaptor-hidden-layer)
        adaptor_hidden_layer="$2"
        shift
        ;;
    *) # unknown option
        echo "Unknown option: $1"
        echo "Usage: $0 [--mode simulation|emulation] [--emulation-seed SEED] [--adaptor-input INPUT] [--adaptor-hidden-layer LAYER]"
        exit 1
        ;;
esac
shift
done

if [ "$MODE" = "simulation" ]; then
    echo "Running in simulation mode..."
    for seed in 10 20 30; do
        python src/simulator/abr_simulator/pensieve/train.py \
            --jump-action \
            --save-dir ${save_dir}/${train_name}/seed_${seed} \
            --exp-name ${train_name} \
            --seed ${seed} \
            --total-epoch ${total_epoch} \
            --video-size-file-dir ${video_size_file_dir} \
            --nagent 10 \
            udr \
            --config-file ${config_file} \
            --val-trace-dir ${val_trace_dir}
    done
elif [ "$MODE" = "emulation" ]; then
    echo "Running in emulation mode..."
    python src/emulator/abr/pensieve/agent_policy/train.py \
        --total-epoch ${total_epoch} \
        --seed ${emulation_seed} \
        --save-dir ${save_dir}/${train_name}_emulation/seed_${emulation_seed} \
        --exp-name ${train_name}_emulation \
        --nagent 10 \
        --video-size-file-dir ${video_size_file_dir} \
        --model-save-interval 10 \
        --original-model-path ${original_model_path} \
        --jump-action \
        --adaptor-input ${adaptor_input} \
        --adaptor-hidden-layer ${adaptor_hidden_layer}
        udr \
        --config-file ${config_file} \
        --val-trace-dir ${val_trace_dir}
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [--mode simulation|emulation] [--emulation-seed SEED] [--adaptor-input INPUT] [--adaptor-hidden-layer LAYER]"
    exit 1
fi
