#!/bin/bash

# setup_conda
# sample command  janechen@node1:~/Genet/src/emulator/abr$ 
# ~/Genet/src/emulator/abr/pensieve/drivers/run_models.sh  ~/Genet/fig_reproduce/model/03_12_model_set_new/ ~/Genet/fig_reproduce/data/synthetic_test_plus_mahimahi_subset/ 03_12_model_summary_subset 6626 0 --use_embedding > /mydata/logs/emu_test_03_12_model_subset.log

MODEL_PARENT_PATH=$1
TRACE_DIR=$2
SUMMARY_DIR_NAME=$3
PORT_ID=$4
AGENT_ID=$5
EXTRA_ARG=$6

# Check if the directory exists
if [ ! -d "$MODEL_PARENT_PATH" ]; then
  echo "Error: Directory does not exist"
 exit 1
fi


# List the subdirectories and file prefixes
echo "Subdirectories and file prefixes in $MODEL_PARENT_PATH:"
for dir in "$MODEL_PARENT_PATH"/*; do
  if [ -d "$dir" ]; then
    subdir_name=$(basename "$dir")

    # Extract the numeric value 'x' from 'server_x'
    if [[ $subdir_name =~ server_([0-9]+) ]]; then
      server_num=${BASH_REMATCH[1]}
      if (( server_num <= 5 )); then
        adaptor_input="ACTION"
      else
        adaptor_input="HIDDEN"
      fi
    else
      echo "Warning: Skipping unexpected directory format: $subdir_name"
      continue
    fi

    file=$(find "$dir" -name "nn_model_ep_[0-9]*.ckpt*" | sort -V | tail -1)
    prefix=$(basename $(echo "$file" | sed 's/nn_model_ep_//; s/\.ckpt.*//'))
    echo " running ${MODEL_PARENT_PATH}$subdir_name/nn_model_ep_${prefix}.ckpt"
    sub_summary_dir=${SUMMARY_DIR_NAME}/${subdir_name}_nn_model_ep_${prefix}
    echo $sub_summary_dir
    mkdir -p  /mydata/results/$sub_summary_dir
    ~/Genet/src/emulator/abr/pensieve/drivers/run_mahimahi_emulation_UDR_3.sh ${MODEL_PARENT_PATH}/$subdir_name/nn_model_ep_${prefix}.ckpt ${TRACE_DIR} ${sub_summary_dir} ${PORT_ID} ${AGENT_ID} ${adaptor_input} ${EXTRA_ARG}
  fi

done