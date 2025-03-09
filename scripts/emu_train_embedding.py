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
    """ SSH into the server and execute the given commands sequentially. """
    print(f"Connecting to {server}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=username)
        for cmd in commands:
            print(f"[{server}] $ {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            # Short delay to ensure commands run sequentially
            time.sleep(1)

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
    """ Execute the training sequence on the remote server. """
    server = server_config["hostname"]
    # For indexing starting at 1, to match the typical IP pattern:
    redis_ip = f"10.10.1.{index + 1}"

    # Dynamically set emulation seed (10 * (node_index + 1))
    emulation_seed = 10 * (index + 1)
    log_filename = f"/mydata/logs/emu_{emulation_seed}.out"

    # The commands to run on each server in sequence:
    commands = [
        # 1) Kill all active tmux sessions
        "tmux kill-server || true",

        # 2) Remove /mydata/*
        "rm -rf /mydata/*",
        "mkdir -p /mydata/logs",  # Recreate logs directory if needed

        # 3) Go to ~/Genet, switch to network-state branch, and pull latest
        "cd ~/Genet && git checkout network-state && git pull",

        # 4) Start bpftrace in a tmux session
        "tmux new-session -d -s bpftrace "
        "'cd ~/Genet/src/emulator/abr/pensieve/virtual_browser/ && sudo bpftrace check.bt > bpftrace_output.txt'",

        # 5) Start Redis in a tmux session
        f"tmux new-session -d -s redis 'redis-server --port {REDIS_PORT} --bind {redis_ip} --protected-mode no'",

        # 6) Replace all occurrences of 10.10.1.1 with the current redis_ip
        f"grep -rl '10.10.1.1' ~/Genet/src/emulator/abr/pensieve/ | xargs sed -i 's/10.10.1.1/{redis_ip}/g'",

        # 7) Start the training in a tmux session with our dynamic emulation seed
        "tmux new-session -d -s training "
        f"'source ~/miniconda/bin/activate genet_env && "
        " cd ~/Genet && "
        f" src/drivers/abr/train_udr3_emu_par.sh --mode emulation --emulation-seed {emulation_seed} "
        f" 2>&1 | tee {log_filename}'"
    ]

    run_remote_commands(server, commands)
    print(f"Training sequence started on {server} (seed={emulation_seed}).")

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
