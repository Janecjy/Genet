#!/bin/bash

# setup_conda
# sample command  janechen@node1:~/Genet/src/emulator/abr$ 
# ~/Genet/src/emulator/abr/pensieve/drivers/run_models.sh <GENET_BASE_PATH> ~/Genet/fig_reproduce/model/03_17_model_set/ ~/Genet/fig_reproduce/data/synthetic_test_plus_mahimahi/ 03_17_model_summary 6626 0 --use_embedding > /mydata/logs/emu_test_03_17_model.log 2>&1

# Usage:
# ./test_script.sh <GENET_BASE_PATH> <MODEL_PARENT_PATH> <TRACE_DIR> <SUMMARY_DIR_NAME> <PORT_ID> <AGENT_ID> <EXTRA_ARG> <SEED> <START_SERVER_ID> <END_SERVER_ID>

GENET_BASE_PATH=$1
MODEL_PARENT_PATH=$2
TRACE_DIR=$3
SUMMARY_DIR_NAME=$4
PORT_ID=$5
AGENT_ID=$6
EXTRA_ARG=$7
SEED=${8:-42}  # Default seed is 42 if not provided
START_SERVER_ID=${9:-1}  # Default start server ID is 1
END_SERVER_ID=${10:-28}  # Default end server ID is 28

# Define the list of (input_type, hidden_size, seed) in order
adaptor_inputs=("original_selection" "hidden_state")
adaptor_hidden_layers=(128 256)
seeds=(10 20 30 40 50)

# Build (input, hidden) pairs
base_combos=()
for input_type in "${adaptor_inputs[@]}"; do
    for hidden_size in "${adaptor_hidden_layers[@]}"; do
        base_combos+=("$input_type:$hidden_size")
    done
done

# Generate full configurations
adaptor_configs=()
for seed in "${seeds[@]}"; do
    for combo in "${base_combos[@]}"; do
        adaptor_configs+=("$combo:$seed")
    done
done

# Set the random seed for consistent trace selection
RANDOM=$SEED

# Check if the model directory exists
if [ ! -d "$MODEL_PARENT_PATH" ]; then
    echo "Error: Directory does not exist"
    exit 1
fi

# Get all trace files and shuffle them deterministically
trace_files=($(ls ${TRACE_DIR}/* | shuf --random-source=<(yes $SEED)))

# Check if MODEL_PARENT_PATH is a direct model directory (single-server mode) or contains server_* subdirectories
model_basename=$(basename "$MODEL_PARENT_PATH")
if [ -f "$MODEL_PARENT_PATH"/*.ckpt* ] 2>/dev/null || [[ ! $model_basename =~ server_ ]] && [ ! -d "$MODEL_PARENT_PATH/server_1" ]; then
    # Single-server mode: MODEL_PARENT_PATH points directly to a model directory
    echo "Single-server mode detected: Using model from $MODEL_PARENT_PATH"
    
    # Use default config for single model
    adaptor_input="original_selection"
    adaptor_hidden_size=128
    
    for trace in "${trace_files[@]}"; do
        echo "Processing trace: $trace"
        temp_trace_dir="/mydata/temp_trace_dir/"
        
        # Create temp directory and copy trace file
        rm -rf "$temp_trace_dir"
        mkdir -p "$temp_trace_dir"
        cp "$trace" "$temp_trace_dir"
        
        # Find latest checkpoint file - check for both patterns
        file=$(find "$MODEL_PARENT_PATH" -maxdepth 1 -name "nn_model_ep_[0-9]*.ckpt*" | sort -V | tail -1)
        
        # If no nn_model_ep files, look for unum_adaptor.ckpt or model_basename.ckpt
        if [ -z "$file" ]; then
            file=$(find "$MODEL_PARENT_PATH" -maxdepth 1 -name "${model_basename}.ckpt*" | sort -V | tail -1)
        fi
        
        if [ -z "$file" ]; then
            echo "No checkpoint file found in $MODEL_PARENT_PATH"
            continue
        fi
        
        # Extract the base filename without extension
        ckpt_basename=$(basename "$file" | sed 's/\.ckpt.*//')
        
        # Construct the checkpoint path to pass to the script
        # Pass only .ckpt (without .data, .index, .meta extensions)
        ckpt_path="${MODEL_PARENT_PATH}/${ckpt_basename}.ckpt"
        
        # Run model with default config
        sub_summary_dir=${SUMMARY_DIR_NAME}/${ckpt_basename}
        mkdir -p "/mydata/results/$sub_summary_dir"
        ${GENET_BASE_PATH}/src/emulator/abr/pensieve/drivers/run_mahimahi_emulation_UDR_3.sh \
            ${GENET_BASE_PATH} \
            ${ckpt_path} \
            ${temp_trace_dir} ${sub_summary_dir} ${PORT_ID} ${AGENT_ID} \
            ${adaptor_input} ${adaptor_hidden_size} 1 ${EXTRA_ARG}
        
        # Cleanup temp trace directory before moving to next trace
        rm -rf "$temp_trace_dir"
    done
else
    # Multi-server mode: MODEL_PARENT_PATH contains server_* subdirectories
    echo "Multi-server mode detected: Processing models in $MODEL_PARENT_PATH"
    
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

                # Extract the numeric server ID
                if [[ $subdir_name =~ server_([0-9]+) ]]; then
                    server_num=${BASH_REMATCH[1]}
                    
                    # Check if the server falls within the requested range
                    if (( server_num < START_SERVER_ID || server_num > END_SERVER_ID )); then
                        continue
                    fi

                    # Assign config using modulo to wrap around if needed
                    config_index=$(( (server_num - 1) % ${#adaptor_configs[@]} ))
                    config=${adaptor_configs[$config_index]}
                    
                    # Extract values from config
                    IFS=':' read -r adaptor_input adaptor_hidden_size seed <<< "$config"

                    echo "Server $server_num assigned config: Input=$adaptor_input, Hidden=$adaptor_hidden_size, Seed=$seed"

                    # Find latest checkpoint file
                    file=$(find "$dir" -name "nn_model_ep_[0-9]*.ckpt*" | sort -V | tail -1)
                    prefix=$(basename "$(echo "$file" | sed 's/nn_model_ep_//; s/\.ckpt.*//')")
                    
                    # Run model with assigned config
                    sub_summary_dir=${SUMMARY_DIR_NAME}/${subdir_name}_nn_model_ep_${prefix}
                    mkdir -p "/mydata/results/$sub_summary_dir"
                    ${GENET_BASE_PATH}/src/emulator/abr/pensieve/drivers/run_mahimahi_emulation_UDR_3.sh \
                        ${GENET_BASE_PATH} \
                        ${MODEL_PARENT_PATH}/$subdir_name/nn_model_ep_${prefix}.ckpt \
                        ${temp_trace_dir} ${sub_summary_dir} ${PORT_ID} ${AGENT_ID} \
                        ${adaptor_input} ${adaptor_hidden_size} 1 ${EXTRA_ARG}
                fi
            fi
        done

        # Cleanup temp trace directory before moving to next trace
        rm -rf "$temp_trace_dir"
    done
fi
