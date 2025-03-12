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
    redis_ip = server_config["redis_ip"]
    # redis_ip = f"10.10.1.{index + 1}"
    emulation_seed = 10 * (index + 1)
    branch = server_config["branch"]
    log_filename = f"/mydata/logs/emu_{emulation_seed}.out"

    commands = [
        # 1) Kill all tmux sessions (clean slate)
        "tmux kill-server || true",

        # 2) Clean up /mydata/*
        "rm -rf /mydata/*",
        "mkdir -p /mydata/logs",

        # 3) Check out network-state branch and pull
        # "cd ~/Genet && git checkout network-state && git pull",
        f"cd ~/Genet && git reset --hard && git fetch && git checkout {branch} && git pull",
        
        # 7) Replace any '10.10.1.1' with redis_ip in .py files only
        f"grep -rl --include='*.py' '10.10.1.1' ~/Genet/src/emulator/abr/pensieve/ | xargs sed -i 's/10.10.1.1/{redis_ip}/g' || true",

        # 4) Create a single main tmux session in detached mode
        "tmux new-session -d -s main 'bash'",

        # 5) In the main session, create a new window for Redis
        "tmux new-window -t main -n redis_window",

        # Send the Redis command to that window
        f"tmux send-keys -t main:redis_window 'redis-server --port {REDIS_PORT} --bind {redis_ip} --protected-mode no' C-m",

        # Wait a bit, then verify Redis is running
        "sleep 3",
        "tmux send-keys -t main:redis_window 'ps aux | grep redis-server' C-m",

        # 6) In the main session, create a new window for bpftrace
        # "tmux new-window -t main -n bpftrace_window",
        # "tmux send-keys -t main:bpftrace_window ",
        # "'cd ~/Genet/src/emulator/abr/pensieve/virtual_browser/ && sudo bpftrace check.bt > bpftrace_output.txt' C-m",

        # 8) In the main session, create a training window
        "tmux new-window -t main -n training_window",

        # Activate conda env, then run the training script in training_window
        f"tmux send-keys -t main:training_window "
        f"'source ~/miniconda/bin/activate genet_env && cd ~/Genet; "
        f"src/drivers/abr/train_udr3_emu_par.sh --mode emulation --emulation-seed {emulation_seed} "
        f"2>&1 | tee {log_filename}' C-m",

        # Optionally list the tmux sessions so we can confirm they've all been created
        "tmux ls"
    ]
    if server_config["run"]:
        run_remote_commands(server, commands)
        print(f"Training sequence started on {server} (seed={emulation_seed}).")

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
