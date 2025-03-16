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

# Define possible configurations
adaptor_inputs = ["original_action_prob", "original_selection", "original_bit_rate", "hidden_state"] #, "raw_state"]
adaptor_hidden_layers = {
    "original_action_prob": [64, 128],
    "original_selection": [64, 128],
    "original_bit_rate": [64, 128],
    "hidden_state": [512, 1024],
    # "raw_state": [512, 1024]
}

# Generate all possible configurations
adaptor_configs = [
    (input_type, hidden_layer)
    for input_type in adaptor_inputs
    for hidden_layer in adaptor_hidden_layers[input_type]
]

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
    emulation_seed = 10 # * (index + 1)
    log_filename = f"/mydata/logs/emu_{emulation_seed}.out"

    if branch == "unum-adaptor":
        adaptor_index = index % len(adaptor_configs)
        adaptor_input, hidden_layer = adaptor_configs[adaptor_index]
        log_filename = f"/mydata/logs/emu_{emulation_seed}_{adaptor_input}_{hidden_layer}.out"
        print(f"Starting training on {server} branch {branch} with adaptor_input={adaptor_input}, hidden_layer={hidden_layer}")
        
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
            f"tmux send-keys -t main:training_window 'src/drivers/abr/train_udr3_emu_par.sh --mode emulation --emulation-seed {emulation_seed} "
            f"--adaptor-input {adaptor_input} --adaptor-hidden-layer {hidden_layer} 2>&1 | tee {log_filename}' C-m",
            "tmux ls"
        ]
        if run:
            run_remote_commands(server, commands)
    else:
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
            f"tmux send-keys -t main:training_window 'src/drivers/abr/train_udr3_emu_par.sh --mode emulation --emulation-seed {emulation_seed} ",
            f"2>&1 | tee {log_filename}' C-m",
            "tmux ls"
        ]
        run_remote_commands(server, commands)

if __name__ == "__main__":
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
