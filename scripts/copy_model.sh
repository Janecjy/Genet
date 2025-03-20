#!/bin/bash

# Ensure configuration files exist
CONFIG_FILE="config.yaml"
TEST_CONFIG_FILE="testconfig.yaml"

if [ ! -f "$CONFIG_FILE" ] || [ ! -f "$TEST_CONFIG_FILE" ]; then
    echo "Error: config.yaml or testconfig.yaml not found!"
    exit 1
fi

# Parse arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <local_destination_directory>"
    exit 1
fi

LOCAL_DEST_DIR="$1"
USERNAME="janechen"  # Your username on all remote nodes

# Extract source servers from config.yaml
source_servers=($(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as file:
    config = yaml.safe_load(file)
print(' '.join([server['hostname'] for server in config['servers']]))
"))

# Extract test nodes from testconfig.yaml
test_nodes=($(python3 -c "
import yaml
with open('$TEST_CONFIG_FILE', 'r') as file:
    config = yaml.safe_load(file)
print(' '.join([server['hostname'] for server in config['test_servers']]))
"))

# Ensure local destination directory exists and clear old models
if [ -d "$LOCAL_DEST_DIR" ]; then
    echo "Removing existing models in $LOCAL_DEST_DIR..."
    rm -rf "$LOCAL_DEST_DIR"
fi
mkdir -p "$LOCAL_DEST_DIR"

# Step 1: Copy models from remote source servers to local directory
for ((i=0; i<${#source_servers[@]}; i++)); do
    server="${source_servers[$i]}"
    n=$((i+1))

    # Define remote directory
    remote_dir="/mydata/results/abr/udr3_emu_par_emulation/seed_*/model_saved/"

    # Check if files exist on the remote server
    file_count=$(ssh "$USERNAME@$server" "find $remote_dir -type f | wc -l")
    if ((file_count == 0)); then
        echo "Warning: No model files found on $server. Skipping..."
        continue
    fi

    # Find the latest model file
    max_N=0
    for file in $(ssh "$USERNAME@$server" "ls $remote_dir/nn_model_ep_*.ckpt* 2>/dev/null"); do
        N=$(echo $file | sed 's/.*ep_//; s/\.ckpt.*//')
        if ((N > max_N)); then
            max_N=$N
        fi
    done

    # Create subdirectories for storage
    mkdir -p "$LOCAL_DEST_DIR/server_$n"

    # Copy the latest model file locally
    for file in $(ssh "$USERNAME@$server" "ls $remote_dir/nn_model_ep_${max_N}.ckpt* 2>/dev/null"); do
        scp "$USERNAME@$server:$file" "$LOCAL_DEST_DIR/server_$n/"
    done

    echo "Copied models from $server to local destination $LOCAL_DEST_DIR/server_$n/"
done

# Step 2: Remove existing models on test nodes before copying
for remote_server in "${test_nodes[@]}"; do
    echo "Removing old models on $remote_server..."
    ssh "$USERNAME@$remote_server" "rm -rf /users/janechen/Genet/fig_reproduce/model/"

    echo "Copying models to $remote_server..."
    scp -r "$LOCAL_DEST_DIR" "$USERNAME@$remote_server:/users/janechen/Genet/fig_reproduce/model/"
    echo "Models copied to $remote_server:/users/janechen/Genet/fig_reproduce/model/"
done

echo "Model copying process completed successfully."
