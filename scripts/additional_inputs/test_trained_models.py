#!/usr/bin/env python3

import paramiko
import time
import yaml
import concurrent.futures
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run testing on trained models from remote servers")
parser.add_argument("--model-parent-path", required=True, help="Path to the parent directory containing model subdirectories")
parser.add_argument("--trace-dir", required=True, help="Directory containing trace files for testing")
parser.add_argument("--summary-dir", required=True, help="Summary directory name for results")
parser.add_argument("--port-id", default="6626", help="Port ID for testing (default: 6626)")
parser.add_argument("--agent-id", default="0", help="Agent ID for testing (default: 0)")
parser.add_argument("--extra-arg", default="--use_embedding", help="Extra arguments for testing")
parser.add_argument("--seed", default="42", help="Random seed for trace selection (default: 42)")
parser.add_argument("--start-server", type=int, default=1, help="Start server ID (default: 1)")
parser.add_argument("--end-server", type=int, default=28, help="End server ID (default: 28)")
args = parser.parse_args()

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers"]
username = "janechen"

# Define the exact configuration that train.py uses
# This must match train.py exactly for consistency

# train.py only uses "original_selection" (not "hidden_state")
adaptor_inputs = ["original_selection"]

# Hidden layer sizes from train.py
adaptor_hidden_layers = [128, 256]

# Seeds from train.py
seeds = [10, 20, 30, 40, 50]

# Context window sizes from train.py
context_windows = [1, 3, 5]

# Build (input, hidden, context_window) combinations exactly as train.py does
base_combos = [
    (input_type, hidden_size, context_window)
    for input_type in adaptor_inputs
    for hidden_size in adaptor_hidden_layers
    for context_window in context_windows
]

# Generate full configurations in the same order as train.py
# seed=10 for all combos, seed=20 for all combos, etc.
adaptor_configs = []
for seed in seeds:
    for (input_type, hidden_layer, context_window) in base_combos:
        adaptor_configs.append((input_type, hidden_layer, context_window, seed))

print(f"Total configurations: {len(adaptor_configs)}")
print("Configuration order:")
for i, config in enumerate(adaptor_configs):
    input_type, hidden_layer, context_window, seed = config
    print(f"  Config {i+1}: Input={input_type}, Hidden={hidden_layer}, ContextWindow={context_window}, Seed={seed}")

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

def test_server(server_config, index):
    """Execute the testing sequence on the remote server."""
    server = server_config["hostname"]
    branch = server_config["branch"]
    redis_ip = server_config["redis_ip"]
    run = server_config["run"]

    # Pick a configuration from adaptor_configs using the same logic as train.py
    config_index = index % len(adaptor_configs)
    adaptor_input, hidden_layer, context_window, emulation_seed = adaptor_configs[config_index]

    log_filename = f"/mydata/logs/test_{emulation_seed}_{adaptor_input}_{hidden_layer}_cw{context_window}.out"
    print(
        f"Starting testing on {server} (branch: {branch}), "
        f"adaptor_input={adaptor_input}, hidden_layer={hidden_layer}, context_window={context_window}, seed={emulation_seed}"
    )

    # Create a unique summary directory name that includes all parameters
    summary_subdir = f"{args.summary_dir}_s{emulation_seed}_{adaptor_input}_h{hidden_layer}_cw{context_window}"

    commands = [
        "tmux kill-server || true",
        "rm -rf /mydata/test_results/*",
        "mkdir -p /mydata/test_results",
        "mkdir -p /mydata/logs",
        f"cd ~/Genet && git reset --hard && git fetch && git checkout {branch} && git pull",
        "tmux new-session -d -s test_main 'bash'",
        "tmux new-window -t test_main -n test_window",
        f"tmux send-keys -t test_main:test_window 'source ~/miniconda/bin/activate genet_env' C-m",
        f"tmux send-keys -t test_main:test_window 'cd ~/Genet' C-m",
        f"tmux send-keys -t test_main:test_window "
        f"'~/Genet/scripts/additional_inputs/test_trained_models.sh "
        f"{args.model_parent_path} {args.trace_dir} {summary_subdir} "
        f"{args.port_id} {args.agent_id} \"{args.extra_arg}\" {args.seed} "
        f"{args.start_server} {args.end_server} 2>&1 | tee {log_filename}' C-m",
        "tmux ls"
    ]
    
    if run:
        run_remote_commands(server, commands)
    else:
        print(f"Skipping {server} (run=false in config)")

def main():
    if not os.path.exists(args.model_parent_path):
        print(f"Error: Model parent path does not exist: {args.model_parent_path}")
        return
    
    if not os.path.exists(args.trace_dir):
        print(f"Error: Trace directory does not exist: {args.trace_dir}")
        return

    print(f"Testing configuration:")
    print(f"  Model parent path: {args.model_parent_path}")
    print(f"  Trace directory: {args.trace_dir}")
    print(f"  Summary directory: {args.summary_dir}")
    print(f"  Server range: {args.start_server} to {args.end_server}")
    print(f"  Port ID: {args.port_id}")
    print(f"  Agent ID: {args.agent_id}")
    print(f"  Extra args: {args.extra_arg}")
    print(f"  Random seed: {args.seed}")

    # Filter servers that should run tests
    test_servers = [s for s in servers if s.get("run", False)]
    
    if not test_servers:
        print("Warning: No servers have 'run' set to true in config.yaml")
        print("Available servers:")
        for i, server in enumerate(servers):
            print(f"  Server {i+1}: {server['hostname']} (run: {server.get('run', False)})")
        return

    print(f"Will run tests on {len(test_servers)} servers")

    # Run testing on servers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(test_servers)) as executor:
        futures = {
            executor.submit(test_server, server_config, i): server_config["hostname"]
            for i, server_config in enumerate(test_servers)
        }
        for future in concurrent.futures.as_completed(futures):
            server = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error starting testing on {server}: {e}")

    print("Testing script commands have been issued to all servers.")

if __name__ == "__main__":
    main()
