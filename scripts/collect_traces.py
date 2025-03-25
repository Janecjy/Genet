import paramiko
import yaml
import argparse
import concurrent.futures

# Constants
USERNAME = "janechen"
GENET_BASE_PATH = "/users/janechen/Genet"
ABR_SCRIPT_PATH = f"{GENET_BASE_PATH}/src/emulator/abr/pensieve/drivers"
VIDEO_SERVER_PATH = f"{GENET_BASE_PATH}/src/emulator/abr/pensieve/video_server"
SYNTHETIC_TRACE_PATH = f"{GENET_BASE_PATH}/abr_trace/training_trace/synthetic_train"

# Load node configuration
CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

nodes = config["servers"]
NUM_NODES = len(nodes)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run distributed data collection on remote nodes")
args = parser.parse_args()

def run_command_blocking(client, cmd, hostname):
    """Runs a command on a remote SSH client and waits for it to finish."""
    print(f"[{hostname}] Executing: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    
    # Block execution until command finishes
    exit_status = stdout.channel.recv_exit_status()
    
    output = stdout.read().decode()
    error = stderr.read().decode()
    
    if exit_status != 0:
        print(f"[{hostname}] ERROR running command: {cmd}\n{error}")
    else:
        print(f"[{hostname}] SUCCESS: {cmd}\n{output}")

def run_remote_commands(node, node_id):
    """SSH into the node, reset the repository, modify redis_ip, and run ABR scripts in a tmux session."""
    hostname = node["hostname"]
    branch = node["branch"]
    redis_ip = node.get("redis_ip", "10.10.1.1")

    print(f"Starting tasks on {hostname}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=USERNAME)

        tmux_session_name = "abr_collection"

        # Commands to set up the environment (run sequentially)
        setup_commands = [
            "tmux kill-server || true",  # Kill any existing tmux sessions
            f"cd {GENET_BASE_PATH} && git reset --hard && git fetch && git checkout {branch} && git pull",
            f"grep -rl --include='*.py' '10.10.1.2' {GENET_BASE_PATH}/src/emulator/abr/pensieve/ | "
            f"xargs sed -i 's/10.10.1.2/{redis_ip}/g' || true",

            # Ensure synthetic trace directory exists
            f"mkdir -p {SYNTHETIC_TRACE_PATH}",
            f"source ~/miniconda/bin/activate genet_env",

            # Generate synthetic traces
            f"bash -c 'source ~/miniconda/bin/activate genet_env && "
            f"cd {GENET_BASE_PATH}/src/emulator/abr && "
            f"python {GENET_BASE_PATH}/src/emulator/abr/pensieve/agent_policy/generate_synthetic_traces.py "
            f"--config-file={GENET_BASE_PATH}/config/abr/udr3_emu_par.json "
            f"--output-dir={SYNTHETIC_TRACE_PATH} --num-traces=10 --seed={node_id * 10}'",

            # Convert real-world traces to Mahimahi format
            f"bash -c 'source ~/miniconda/bin/activate genet_env && "
            f"cd {GENET_BASE_PATH}/scripts && "
            f"python {GENET_BASE_PATH}/scripts/convert_to_mahimahi.py --input-dir {GENET_BASE_PATH}/abr_trace/training_trace/fcc_train "
            f"--output-dir {GENET_BASE_PATH}/abr_trace/training_trace/fcc_train_mahimahi --node-id {node_id} --num-nodes {NUM_NODES}'",

            f"bash -c 'source ~/miniconda/bin/activate genet_env && "
            f"cd {GENET_BASE_PATH}/scripts && "
            f"python {GENET_BASE_PATH}/scripts/convert_to_mahimahi.py --input-dir {GENET_BASE_PATH}/abr_trace/training_trace/norway_train "
            f"--output-dir {GENET_BASE_PATH}/abr_trace/training_trace/norway_train_mahimahi --node-id {node_id} --num-nodes {NUM_NODES}'",
        ]

        # **Run setup commands one by one, waiting for each to complete**
        for cmd in setup_commands:
            run_command_blocking(client, cmd, hostname)

        # **Start Video Server in its own tmux session**
        print(f"[{hostname}] Starting video server...")
        run_command_blocking(client, "tmux new-session -d -s video_server", hostname)
        run_command_blocking(client, f"tmux send-keys -t video_server 'source ~/miniconda/bin/activate genet_env' C-m", hostname)
        run_command_blocking(client, f"tmux send-keys -t video_server 'cd {VIDEO_SERVER_PATH}' C-m", hostname)
        run_command_blocking(client, f"tmux send-keys -t video_server 'python video_server.py --port 6626' C-m", hostname)

        # **Start a tmux session for ABR collection**
        print(f"[{hostname}] Starting ABR collection in tmux...")
        run_command_blocking(client, f"tmux new-session -d -s {tmux_session_name}", hostname)
        run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} 'source ~/miniconda/bin/activate genet_env' C-m", hostname)
        run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} 'cd ~/Genet/src/emulator/abr' C-m", hostname)
        run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} 'chmod +x {ABR_SCRIPT_PATH}/run_mahimahi_emulation_BBA.sh' C-m", hostname)
        run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} 'chmod +x {ABR_SCRIPT_PATH}/run_mahimahi_emulation_MPC.sh' C-m", hostname)

        # List of ABR collection commands to run sequentially
        abr_commands = [
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_BBA.sh {GENET_BASE_PATH}/abr_trace/training_trace/fcc_train_mahimahi/ 03_24_collect 6626 0 --collection",
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_BBA.sh {GENET_BASE_PATH}/abr_trace/training_trace/norway_train_mahimahi/ 03_24_collect 6626 0 --collection",
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_BBA.sh {GENET_BASE_PATH}/abr_trace/training_trace/synthetic_train/ 03_24_collect 6626 0 --collection",
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_MPC.sh {GENET_BASE_PATH}/abr_trace/training_trace/fcc_train_mahimahi/ 03_24_collect 6626 0 --collection",
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_MPC.sh {GENET_BASE_PATH}/abr_trace/training_trace/norway_train_mahimahi/ 03_24_collect 6626 0 --collection",
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_MPC.sh {GENET_BASE_PATH}/abr_trace/training_trace/synthetic_train/ 03_24_collect 6626 0 --collection",
        ]

        # **Execute ABR scripts in tmux, ensuring sequential execution**
        for cmd in abr_commands:
            run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} '{cmd}' C-m", hostname)
            run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} 'wait' C-m", hostname)

        print(f"[{hostname}] Tasks started in tmux session: {tmux_session_name}")

    except Exception as e:
        print(f"[{hostname}] Error: {e}")

    finally:
        client.close()

# Run on all nodes in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
    futures = {executor.submit(run_remote_commands, node, node_id + 1): node["hostname"] for node_id, node in enumerate(nodes)}

    for future in concurrent.futures.as_completed(futures):
        hostname = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"[{hostname}] Error: {e}")

print("All remote tasks started successfully.")
