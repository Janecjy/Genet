#!/bin/bash

# Test script for models trained by train.py
# This script tests the configurations that match what train.py actually trains
# Usage:
# ./test_trained_models.sh <MODEL_PARENT_PATH> <TRACE_DIR> <SUMMARY_DIR_NAME> <PORT_ID> <AGENT_ID> <EXTRA_ARG> <SEED> <START_SERVER_ID> <END_SERVER_ID>

MODEL_PARENT_PATH=$1
TRACE_DIR=$2
SUMMARY_DIR_NAME=$3
PORT_ID=$4
AGENT_ID=$5
EXTRA_ARG=$6
SEED=${7:-42}  # Default seed is 42 if not provided
START_SERVER_ID=${8:-1}  # Default start server ID is 1
END_SERVER_ID=${9:-20}  # Default end server ID is 20 (actual number of servers)

# Define the exact configuration that train.py uses
# train.py only uses "original_selection" (not "hidden_state")
adaptor_inputs=("original_selection")

# Hidden layer sizes from train.py
adaptor_hidden_layers=(128 256)

# Seeds from train.py
seeds=(10 20 30 40 50)

# Context window sizes from train.py
context_windows=(1 3 5)

# Build (input, hidden, context_window) combinations exactly as train.py does
base_combos=()
for input_type in "${adaptor_inputs[@]}"; do
    for hidden_size in "${adaptor_hidden_layers[@]}"; do
        for context_window in "${context_windows[@]}"; do
            base_combos+=("$input_type:$hidden_size:$context_window")
        done
    done
done

# Generate full configurations in the same order as train.py
# seed=10 for all combos, seed=20 for all combos, etc.
adaptor_configs=()
for seed in "${seeds[@]}"; do
    for combo in "${base_combos[@]}"; do
        adaptor_configs+=("$combo:$seed")
    done
done

echo "Total configurations: ${#adaptor_configs[@]}"
echo "Configuration order:"
for i in "${!adaptor_configs[@]}"; do
    echo "  Config $((i+1)): ${adaptor_configs[$i]}"
done

# Set the random seed for consistent trace selection
RANDOM=$SEED

# Check if the model directory exists
if [ ! -d "$MODEL_PARENT_PATH" ]; then
    echo "Error: Directory does not exist: $MODEL_PARENT_PATH"
    exit 1
fi

# Check if trace directory exists
if [ ! -d "$TRACE_DIR" ]; then
    echo "Error: Trace directory does not exist: $TRACE_DIR"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p /mydata/logs

# Get all trace files and shuffle them deterministically
trace_files=($(ls ${TRACE_DIR}/* | shuf --random-source=<(yes $SEED)))

echo "Found ${#trace_files[@]} trace files"
echo "Processing servers from $START_SERVER_ID to $END_SERVER_ID"

for trace in "${trace_files[@]}"; do
    echo "Processing trace: $trace"
    temp_trace_dir="/mydata/temp_trace_dir/"
    
    # Create temp directory and copy trace file
    rm -rf "$temp_trace_dir"
    mkdir -p "$temp_trace_dir"
    cp "$trace" "$temp_trace_dir"

    # Process all models in the selected server ID range
    for dir in "$MODEL_PARENT_PATH"/*; do
        if [ -d "$dir" ]; then
            subdir_name=$(basename "$dir")

            # Extract the numeric server ID (assuming format like server_X)
            if [[ $subdir_name =~ server_([0-9]+) ]]; then
                server_num=${BASH_REMATCH[1]}
                
                # Check if the server falls within the requested range
                if (( server_num < START_SERVER_ID || server_num > END_SERVER_ID )); then
                    echo "Skipping server $server_num (outside range $START_SERVER_ID-$END_SERVER_ID)"
                    continue
                fi

                # Assign config using modulo to wrap around if needed
                # This matches the logic in train.py: config_index = index % len(adaptor_configs)
                config_index=$(( (server_num - 1) % ${#adaptor_configs[@]} ))
                config=${adaptor_configs[$config_index]}
                
                # Extract values from config
                IFS=':' read -r adaptor_input adaptor_hidden_size context_window seed <<< "$config"

                echo "Server $server_num assigned config: Input=$adaptor_input, Hidden=$adaptor_hidden_size, ContextWindow=$context_window, Seed=$seed"

                # Find latest checkpoint file
                file=$(find "$dir" -name "nn_model_ep_[0-9]*.ckpt*" | sort -V | tail -1)
                
                if [ -z "$file" ]; then
                    echo "Warning: No checkpoint files found in $dir"
                    continue
                fi
                
                prefix=$(basename "$(echo "$file" | sed 's/nn_model_ep_//; s/\.ckpt.*//')")
                
                # Create unique summary directory that includes all config parameters
                sub_summary_dir=${SUMMARY_DIR_NAME}/${subdir_name}_${adaptor_input}_h${adaptor_hidden_size}_cw${context_window}_s${seed}_ep_${prefix}
                mkdir -p "/mydata/results/$sub_summary_dir"
                
                echo "Running test with checkpoint: $file"
                echo "Summary directory: /mydata/results/$sub_summary_dir"
                
                # Run model with assigned config
                # Note: The emulation script now supports context window parameter
                ~/Genet/src/emulator/abr/pensieve/drivers/run_mahimahi_emulation_UDR_3.sh \
                    ${MODEL_PARENT_PATH}/$subdir_name/nn_model_ep_${prefix}.ckpt \
                    ${temp_trace_dir} ${sub_summary_dir} ${PORT_ID} ${AGENT_ID} \
                    ${adaptor_input} ${adaptor_hidden_size} ${context_window} ${EXTRA_ARG}
                
                echo "Completed testing server $server_num"
            else
                echo "Warning: Directory name '$subdir_name' doesn't match expected pattern 'server_X'"
            fi
        fi
    done

    # Cleanup temp trace directory before moving to next trace
    rm -rf "$temp_trace_dir"
done

echo "Testing completed for all traces and servers in range $START_SERVER_ID-$END_SERVER_ID"
