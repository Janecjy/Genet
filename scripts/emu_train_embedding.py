#!/usr/bin/env python3

import paramiko
import time
import yaml
import concurrent.futures
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run training on remote servers")
args = parser.parse_args()

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers"]
username = "janechen"

# Define the custom Redis port
REDIS_PORT = 2666

# 2) Restrict to only original_selection and hidden_state
adaptor_inputs = ["original_selection", "hidden_state"]

# 3) Use hidden-layer sizes of 128 and 256
adaptor_hidden_layers = [128, 256]

# 4) Multiple seeds, but we prioritize unique (input, hidden) before changing seeds
seeds = [10, 20, 30, 40, 50]

# First build the (input, hidden) pairs (no seed)
base_combos = [
    (input_type, hidden_size)
    for input_type in adaptor_inputs
    for hidden_size in adaptor_hidden_layers
]
# base_combos = [
#   ("original_selection", 128),
#   ("original_selection", 256),
#   ("hidden_state", 128),
#   ("hidden_state", 256),
# ]

# Then create a final list of (input_type, hidden_size, seed) in the order:
# seed=10 for all combos, seed=20 for all combos, ...
adaptor_configs = []
for seed in seeds:
    for (input_type, hidden_layer) in base_combos:
        adaptor_configs.append((input_type, hidden_layer, seed))

def run_remote_commands(server, commands):
    """SSH into the server and execute the given commands sequentially."""
    print(f"Connecting to {server}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=username)
        for cmd in commands:
            print(f"[{server}] $ {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            time.sleep(1)  # brief pause to let commands run

            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()

            if out:
                print(f"[{server}] STDOUT:\n{out}")
            if err:
                print(f"[{server}] STDERR:\n{err}")

    except Exception as e:
        print(f"Error running commands on {server}: {e}")
    finally:
        client.close()

def train_server(server_config, index):
    """Execute the training sequence on the remote server."""
    server = server_config["hostname"]
    branch = server_config["branch"]
    redis_ip = server_config["redis_ip"]
    run = server_config["run"]

    # Pick a configuration from adaptor_configs
    # If we have more combos than servers, only the first N combos are used.
    config_index = index % len(adaptor_configs)
    adaptor_input, hidden_layer, emulation_seed = adaptor_configs[config_index]

    log_filename = f"/mydata/logs/emu_{emulation_seed}_{adaptor_input}_{hidden_layer}.out"
    print(
        f"Starting training on {server} (branch: {branch}), "
        f"adaptor_input={adaptor_input}, hidden_layer={hidden_layer}, seed={emulation_seed}"
    )

    if branch == "unum-adaptor":
        commands = [
            "tmux kill-server || true",
            "rm -rf /mydata/*",
            "mkdir -p /mydata/logs",
            f"cd ~/Genet && git reset --hard && git fetch && git checkout {branch} && git pull",
            f"grep -rl --include='*.py' '10.10.1.2' ~/Genet/src/emulator/abr/pensieve/ | xargs sed -i 's/10.10.1.2/{redis_ip}/g' || true",
            "tmux new-session -d -s main 'bash'",
            "tmux new-window -t main -n training_window",
            f"tmux send-keys -t main:training_window 'source ~/miniconda/bin/activate genet_env' C-m",
            f"tmux send-keys -t main:training_window 'cd ~/Genet' C-m",
            f"tmux send-keys -t main:training_window "
            f"'src/drivers/abr/train_udr3_emu_par.sh --mode emulation --emulation-seed {emulation_seed} "
            f"--adaptor-input {adaptor_input} --adaptor-hidden-layer {hidden_layer} 2>&1 | tee {log_filename}' C-m",
            "tmux ls"
        ]
        if run:
            run_remote_commands(server, commands)
    else:
        # Fallback if branch != "unum-adaptor"
        log_filename = f"/mydata/logs/emu_{emulation_seed}.out"
        commands = [
            "tmux kill-server || true",
            "rm -rf /mydata/*",
            "mkdir -p /mydata/logs",
            "cd ~/Genet && git reset --hard && git pull",
            "tmux new-session -d -s main 'bash'",
            f"grep -rl --include='*.py' '10.10.1.1' ~/Genet/src/emulator/abr/pensieve/ | xargs sed -i 's/10.10.1.1/{redis_ip}/g' || true",
            "tmux new-window -t main -n training_window",
            f"tmux send-keys -t main:training_window 'source ~/miniconda/bin/activate genet_env' C-m",
            f"tmux send-keys -t main:training_window 'cd ~/Genet' C-m",
            f"tmux send-keys -t main:training_window "
            f"'src/drivers/abr/train_udr3_emu_par.sh --mode emulation --emulation-seed {emulation_seed} "
            f"2>&1 | tee {log_filename}' C-m",
            "tmux ls"
        ]
        run_remote_commands(server, commands)

if __name__ == "__main__":
    # Run training on all servers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
        futures = {
            executor.submit(train_server, server_config, i): server_config["hostname"]
            for i, server_config in enumerate(servers)
        }
        for future in concurrent.futures.as_completed(futures):
            server = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error starting training on {server}: {e}")

    print("Training script commands have been issued to all servers.")
