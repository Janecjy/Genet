#!/bin/bash

# Ensure test configuration file exists
TEST_CONFIG_FILE="testconfig.yaml"

if [ ! -f "$TEST_CONFIG_FILE" ]; then
    echo "Error: testconfig.yaml not found!"
    exit 1
fi

# Parse arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <local_results_directory>"
    exit 1
fi

LOCAL_RESULTS_DIR="$1"
USERNAME="janechen"  # Your username on all remote nodes

# Extract test nodes from testconfig.yaml and map branches
test_nodes=($(python3 -c "
import yaml
with open('$TEST_CONFIG_FILE', 'r') as file:
    config = yaml.safe_load(file)
print('\n'.join([f'{server[\"hostname\"]}:{server[\"branch\"]}' for server in config['test_servers']]))
"))

# Ensure local results directory exists
mkdir -p "$LOCAL_RESULTS_DIR"

# Step 1: Collect results from test nodes
for node_branch in "${test_nodes[@]}"; do
    IFS=":" read -r remote_server branch <<< "$node_branch"

    echo "Processing $remote_server with branch: $branch"

    if [ "$branch" == "sim-reproduce" ]; then
        REMOTE_RESULTS_DIR="/users/janechen/Genet/src/emulator/abr/pensieve/tests/UDR-3_60_40_/users/janechen/Genet/abr_trace/testing_trace_mahimahi/"
        LOCAL_DEST_DIR="$LOCAL_RESULTS_DIR/pensieve-original"

    elif [ "$branch" == "unum-adaptor" ]; then
        SUMMARY_DIR=$(ssh "$USERNAME@$remote_server" "ls /mydata/results/ | grep -m1 '$LOCAL_RESULTS_DIR'" || echo "default_summary")

        if [ -z "$SUMMARY_DIR" ] || [ "$SUMMARY_DIR" == "default_summary" ]; then
            echo "Warning: No summary directory found on $remote_server. Skipping..."
            continue
        fi

        REMOTE_RESULTS_DIR="/mydata/results/$SUMMARY_DIR/"
        LOCAL_DEST_DIR="$LOCAL_RESULTS_DIR/$SUMMARY_DIR"

    else
        echo "Skipping unknown branch type on $remote_server"
        continue
    fi

    # Ensure local destination directory exists
    mkdir -p "$LOCAL_DEST_DIR"

    echo "Collecting results from $remote_server ($branch) into $LOCAL_DEST_DIR..."
    scp -r "$USERNAME@$remote_server:$REMOTE_RESULTS_DIR" "$LOCAL_DEST_DIR"

    if [ $? -eq 0 ]; then
        echo "Results from $remote_server saved to $LOCAL_DEST_DIR"
    else
        echo "Error copying results from $remote_server"
    fi
done

echo "Test results collection completed successfully."
