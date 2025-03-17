#!/bin/bash

# setup_conda
# sample command  janechen@node1:~/Genet/src/emulator/abr$ 
# ~/Genet/src/emulator/abr/pensieve/drivers/run_models.sh  ~/Genet/fig_reproduce/model/03_17_model_set/ ~/Genet/fig_reproduce/data/synthetic_test_plus_mahimahi/ 03_17_model_summary_subset 6626 0 --use_embedding > /mydata/logs/emu_test_03_17_model_subset.log

MODEL_PARENT_PATH=$1
TRACE_DIR=$2
SUMMARY_DIR_NAME=$3
PORT_ID=$4
AGENT_ID=$5
EXTRA_ARG=$6

# Define possible configurations
adaptor_inputs=("original_action_prob" "original_selection" "original_bit_rate" "hidden_state")
adaptor_hidden_layers=(64 128 64 128 64 128 512 1024)

# Check if the directory exists
if [ ! -d "$MODEL_PARENT_PATH" ]; then
  echo "Error: Directory does not exist"
  exit 1
fi

# List all trace files
trace_files=($(ls ${TRACE_DIR}/*))

for trace in "${trace_files[@]}"; do
    echo "Processing trace: $trace"
    temp_trace_dir="/mydata/temp_trace_dir"
    
    # Create temp directory and copy trace file
    rm -rf "$temp_trace_dir"
    mkdir -p "$temp_trace_dir"
    cp "$trace" "$temp_trace_dir"

    # Process all models with the current trace
    for dir in "$MODEL_PARENT_PATH"/*; do
      if [ -d "$dir" ]; then
        subdir_name=$(basename "$dir")

        # Extract the numeric value 'x' from 'server_x'
        if [[ $subdir_name =~ server_([0-9]+) ]]; then
          server_num=${BASH_REMATCH[1]}
          if (( server_num >= 1 && server_num <= 8 )); then
            adaptor_input=${adaptor_inputs[$(((server_num - 1) / 2))]}
            adaptor_hidden_size=${adaptor_hidden_layers[$((server_num - 1))]}
          elif (( server_num >= 9 && server_num <= 13 )); then
            adaptor_input="ACTION"
            adaptor_hidden_size=128
          elif (( server_num >= 14 && server_num <= 18 )); then
            adaptor_input="HIDDEN"
            adaptor_hidden_size=128
          else
            echo "Warning: Skipping unexpected directory format: $subdir_name"
            continue
          fi
        fi

        file=$(find "$dir" -name "nn_model_ep_[0-9]*.ckpt*" | sort -V | tail -1)
        prefix=$(basename $(echo "$file" | sed 's/nn_model_ep_//; s/\.ckpt.*//'))
        echo " running ${MODEL_PARENT_PATH}$subdir_name/nn_model_ep_${prefix}.ckpt"
        sub_summary_dir=${SUMMARY_DIR_NAME}/${subdir_name}_nn_model_ep_${prefix}
        echo $sub_summary_dir
        mkdir -p  /mydata/results/$sub_summary_dir
        ~/Genet/src/emulator/abr/pensieve/drivers/run_mahimahi_emulation_UDR_3.sh ${MODEL_PARENT_PATH}/$subdir_name/nn_model_ep_${prefix}.ckpt ${temp_trace_dir} ${sub_summary_dir} ${PORT_ID} ${AGENT_ID} ${adaptor_input} ${adaptor_hidden_size} ${EXTRA_ARG}
      fi
    done
    
    # Cleanup temp trace directory before moving to next trace
    rm -rf "$temp_trace_dir"
done
