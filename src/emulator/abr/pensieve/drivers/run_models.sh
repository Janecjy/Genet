#!/bin/bash

# sample command  janechen@node1:~/Genet/src/emulator/abr$ ~/Genet/src/emulator/abr/pensieve/drivers/run_models.sh  ~/Genet/fig_reproduce/model/03_10_model_set/ ~/Genet/fig_reproduce/data/tmp/ 03_10_model_summary 6626 0 --use_embedding > /mydata/logs/emu_test_03_10_model.log

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
    file=$(find "$dir" -name "nn_model_ep_[0-9]*.ckpt*" | sort -V | tail -1)
    prefix=$(basename $(echo "$file" | sed 's/nn_model_ep_//; s/\.ckpt.*//'))
    echo " running ${MODEL_PARENT_PATH}$subdir_name/nn_model_ep_${prefix}.ckpt"
    sub_summary_dir=${SUMMARY_DIR_NAME}/${subdir_name}_nn_model_ep_${prefix}
    echo $sub_summary_dir
    mkdir -p  ~/Genet/src/emulator/abr/pensieve/tests/$sub_summary_dir
    ~/Genet/src/emulator/abr/pensieve/drivers/run_mahimahi_emulation_UDR_3.sh ${MODEL_PARENT_PATH}/$subdir_name/nn_model_ep_${prefix}.ckpt ${TRACE_DIR} ${sub_summary_dir} ${PORT_ID} ${AGENT_ID} ${EXTRA_ARG}
  fi

done