#!/usr/bin/env python3

import paramiko
import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
import sys

# Load server addresses from config.yaml
CONFIG_FILE = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

# Extract server hostnames
servers = [server["hostname"] for server in config["servers"]]

username = "janechen"
REMOTE_LOG_DIR = "/mydata/results/abr/udr3_emu_par_emulation/"
LOCAL_PLOT_DIR = "./training_plots/"
REMOTE_PLOT_DIR = "/mydata/plots/"

os.makedirs(LOCAL_PLOT_DIR, exist_ok=True)

def plot_training_curve(data, server_idx):
    """Plot loss, avg_reward, and avg_entropy curves for a node and save the figure."""
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(data["epoch"], data["loss"], marker=".", linestyle="-", label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - Server {server_idx}")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(data["epoch"], data["avg_reward"], marker=".", linestyle="-", label="Avg Reward", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Reward")
    plt.title(f"Average Reward - Server {server_idx}")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(data["epoch"], data["avg_entropy"], marker=".", linestyle="-", label="Avg Entropy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Entropy")
    plt.title(f"Average Entropy - Server {server_idx}")
    plt.grid(True)

    plt.tight_layout()
    filename = f"{LOCAL_PLOT_DIR}server_{server_idx}.png"
    plt.savefig(filename)
    plt.close()

def fetch_and_plot(server, server_idx):
    """SSH into the server, read logs, plot graphs, and fetch plots back."""
    print(f"Processing {server} as Server {server_idx}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=username)
        sftp = client.open_sftp()

        # Find log_train file in each seed directory
        stdin, stdout, stderr = client.exec_command(f"ls -d {REMOTE_LOG_DIR}*/log_train 2>/dev/null")
        log_files = stdout.read().decode().strip().split("\n")

        for log_file in log_files:
            if not log_file.strip():
                continue

            print(f"Processing log: {log_file}")

            # Read the log file
            with sftp.open(log_file, "r") as f:
                lines = f.readlines()

            # Parse log data
            data = {"epoch": [], "loss": [], "avg_reward": [], "avg_entropy": []}
            for line in lines[1:]:  # Skip header
                values = line.split()
                data["epoch"].append(int(values[0]))
                data["loss"].append(float(values[1]))
                data["avg_reward"].append(float(values[2]))
                data["avg_entropy"].append(float(values[3]))

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Generate plot
            plot_training_curve(df, server_idx)

            # Create remote directory for plots if not exists
            client.exec_command(f"mkdir -p {REMOTE_PLOT_DIR}")

            # Upload plot to the remote server
            remote_plot_path = f"{REMOTE_PLOT_DIR}server_{server_idx}.png"
            local_plot_path = f"{LOCAL_PLOT_DIR}server_{server_idx}.png"
            sftp.put(local_plot_path, remote_plot_path)

            # SCP the plot back to local machine
            sftp.get(remote_plot_path, local_plot_path)

            print(f"Plot saved: {local_plot_path}")

        sftp.close()

    except Exception as e:
        print(f"Error processing {server}: {e}")

    finally:
        client.close()

# Process all servers in order
for i, server in enumerate(servers):
    fetch_and_plot(server, i + 1)

print("All plots have been fetched successfully.")
