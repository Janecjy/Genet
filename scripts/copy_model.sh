#!/bin/bash

# Ensure config.yaml is provided
CONFIG_FILE="config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.yaml not found!"
    exit 1
fi

# Extract the server list from config.yaml
model_server_list=($(yq e '.servers[].hostname' "$CONFIG_FILE"))

# Define the directory
remote_dir=$1

# Loop through each server
for ((i=0; i<${#model_server_list[@]}; i++)); do
    server="janechen@${model_server_list[$i]}"
    n=$((i+1))

    # Define the local directory using seed_*/model_saved pattern
    local_dir="/mydata/results/abr/udr3_emu_par_emulation/seed_*/model_saved/"

    # Check if any matching directory exists
    file_count=$(ssh $server "find /mydata/results/abr/udr3_emu_par_emulation/seed_*/model_saved -type f | wc -l")
    if ((file_count == 0)); then
        echo "Warning: No model files found in $local_dir on server $server. Skipping..."
        continue
    fi

    # Find the set of files with the biggest N
    max_N=0
    for file in $(ssh $server "ls /mydata/results/abr/udr3_emu_par_emulation/seed_*/model_saved/nn_model_ep_*.ckpt* 2>/dev/null"); do
        N=$(echo $file | sed 's/.*ep_//; s/\.ckpt.*//')
        if ((N > max_N)); then
            max_N=$N
        fi
    done

    # Create a subdirectory for each server on the remote server
    mkdir -p ~/$remote_dir/server_$n
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create directory ~/$remote_dir/server_$n on remote server"
        continue
    fi

    # Copy the files with the biggest N to the remote server
    for file in $(ssh $server "ls /mydata/results/abr/udr3_emu_par_emulation/seed_*/model_saved/nn_model_ep_${max_N}.ckpt* 2>/dev/null"); do
        scp $server:$file ~/$remote_dir/server_$n/
    done
done
