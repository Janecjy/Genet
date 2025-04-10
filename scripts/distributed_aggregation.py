import paramiko
import yaml
import concurrent.futures
import argparse
import time
import os
import subprocess

# Parse arguments
parser = argparse.ArgumentParser(description="Run or collect RTT dataset sampling from remote nodes.")
parser.add_argument("--collect", action="store_true", help="Collect generated real/synthetic .p files via scp")
parser.add_argument("--target-path", type=str, default="mll:/datastor1/janec/rtt", help="Where to copy collected files")
args = parser.parse_args()

# Load config
CONFIG_FILE = "testconfig.yaml"
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers"]
username = "janechen"
script_name = "rtt_aggregation.py"
local_script_path = f"/Users/janechen/Desktop/ccBench/{script_name}"
remote_script_path = f"~/ccBench/{script_name}"
output_prefix = "/mydata/pensieve_rtt_node"

def run_remote_sampler(server, idx):
    hostname = server["hostname"]
    print(f"[{hostname}] Launching RTT sampler in tmux...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=username)

        commands = [
            "tmux kill-session -t samplertt || true",
            "mkdir -p /mydata",
            "tmux new-session -d -s samplertt",
            f"tmux send-keys -t samplertt 'cd ~/Genet' C-m",
            f"tmux send-keys -t samplertt 'git pull' C-m",
            f"tmux send-keys -t samplertt 'source ~/miniconda/bin/activate genet_env' C-m",
            f"tmux send-keys -t samplertt 'cd scripts' C-m",
            f"tmux send-keys -t samplertt 'python {script_name}' C-m"
        ]

        for cmd in commands:
            print(f"[{hostname}] $ {cmd}")
            client.exec_command(cmd)
            time.sleep(0.3)

        print(f"[{hostname}] ✅ Sampler launched.")

    except Exception as e:
        print(f"[{hostname}] ❌ Error: {e}")
    finally:
        client.close()

def collect_node_outputs(server, idx):
    hostname = server["hostname"]
    base = f"{output_prefix}{idx}"
    files = [
        (f"{username}@{hostname}:{base}_real.p", f"{args.target_path}/pensieve_rtt_node{idx}_real.p"),
        (f"{username}@{hostname}:{base}_synthetic.p", f"{args.target_path}/pensieve_rtt_node{idx}_synthetic.p"),
    ]

    for remote, local in files:
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-3", remote, local
        ]
        print(f"[{hostname}] SCP {remote} → {local}")
        try:
            subprocess.run(scp_cmd, check=True)
            print(f"[{hostname}] ✅ Collected {os.path.basename(local)}")
        except subprocess.CalledProcessError:
            print(f"[{hostname}] ❌ Failed to collect {os.path.basename(local)}")

if __name__ == "__main__":
    if args.collect:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
            futures = {
                executor.submit(collect_node_outputs, server, idx): server["hostname"]
                for idx, server in enumerate(servers)
            }
            for future in concurrent.futures.as_completed(futures):
                hostname = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[{hostname}] ❌ SCP failed: {e}")
        print("✅ Dataset collection complete.")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
            futures = {
                executor.submit(run_remote_sampler, server, idx): server["hostname"]
                for idx, server in enumerate(servers)
            }
            for future in concurrent.futures.as_completed(futures):
                hostname = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[{hostname}] ❌ Failed to launch: {e}")
        print("✅ Dataset generation launched on all nodes.")
