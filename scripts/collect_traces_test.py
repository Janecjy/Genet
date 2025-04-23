import paramiko
import yaml
import argparse
import concurrent.futures
import os

# Constants
USERNAME = "janechen"
GENET_BASE_PATH = "/users/janechen/Genet"
ABR_SCRIPT_PATH = f"{GENET_BASE_PATH}/src/emulator/abr/pensieve/drivers"
VIDEO_SERVER_PATH = f"{GENET_BASE_PATH}/src/emulator/abr/pensieve/video_server"
# SYNTHETIC_TRACE_PATH = f"{GENET_BASE_PATH}/abr_trace/training_trace/synthetic_train"
TMP_TRACE_PATH = "/mydata/tmp_traces"

# Load node configuration
CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

nodes = config["servers"]
NUM_NODES = len(nodes)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run distributed data collection on remote nodes")
args = parser.parse_args()

def scp_to_remote(client, local_dir, remote_dir, hostname):
    """Recursively copies a local directory to the remote directory via SFTP."""
    sftp = client.open_sftp()
    for root, _, files in os.walk(local_dir):
        remote_path = os.path.join(remote_dir, os.path.relpath(root, local_dir))
        try:
            sftp.mkdir(remote_path)
        except IOError:
            pass  # directory may already exist
        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file)
            print(f"[{hostname}] Uploading {local_file} -> {remote_file}")
            sftp.put(local_file, remote_file)
    sftp.close()


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
            f"grep -rl --include='*.py' 'Checkpoint-Combined_10RTT_6col_Transformer3_256_4_4_32_4_lr_0.0001_boundaries-quantile50-merged_multi-559iter.p' {GENET_BASE_PATH}/src/emulator/abr/pensieve/ | "
            f"xargs sed -i 's|Checkpoint-Combined_10RTT_6col_Transformer3_256_4_4_32_4_lr_0.0001_boundaries-quantile50-merged_multi-559iter.p|Checkpoint-Combined_10RTT_6col_Transformer3_256_4_4_32_4_lr_0.0001_boundaries-quantile50-merged_multi-509iter.p|g' || true"


            # Ensure synthetic trace directory exists
            # f"mkdir -p {SYNTHETIC_TRACE_PATH}",
            f"mkdir -p /mydata/logs",
            f"source ~/miniconda/bin/activate genet_env",

            # # Generate synthetic traces
            # f"bash -c 'source ~/miniconda/bin/activate genet_env && "
            # f"cd {GENET_BASE_PATH}/src/emulator/abr && "
            # f"python {GENET_BASE_PATH}/src/emulator/abr/pensieve/agent_policy/generate_synthetic_traces.py "
            # f"--config-file={GENET_BASE_PATH}/config/abr/udr3_emu_par.json "
            # f"--output-dir={SYNTHETIC_TRACE_PATH} --num-traces=10 --seed={node_id * 10}'",

            # Convert real-world traces to Mahimahi format
            # f"bash -c 'source ~/miniconda/bin/activate genet_env && "
            # f"cd {GENET_BASE_PATH}/scripts && "
            # f"python {GENET_BASE_PATH}/scripts/convert_to_mahimahi.py --input-dir {GENET_BASE_PATH}/abr_trace/tes/fcc_train "
            # f"--output-dir {GENET_BASE_PATH}/abr_trace/training_trace/fcc_train_mahimahi --node-id {node_id} --num-nodes {NUM_NODES}'",

            # f"bash -c 'source ~/miniconda/bin/activate genet_env && "
            # f"cd {GENET_BASE_PATH}/scripts && "
            # f"python {GENET_BASE_PATH}/scripts/convert_to_mahimahi.py --input-dir {GENET_BASE_PATH}/abr_trace/training_trace/norway_train "
            # f"--output-dir {GENET_BASE_PATH}/abr_trace/training_trace/norway_train_mahimahi --node-id {node_id} --num-nodes {NUM_NODES}'",
            
            # Ensure temporary trace directory exists
            f"mkdir -p {TMP_TRACE_PATH}",

            # # Copy trace files to the temporary directory
            # f"cp {GENET_BASE_PATH}/abr_trace/training_trace/fcc_train_mahimahi/* {TMP_TRACE_PATH}/",
            # f"cp {GENET_BASE_PATH}/abr_trace/training_trace/norway_train_mahimahi/* {TMP_TRACE_PATH}/",
            # f"cp {GENET_BASE_PATH}/abr_trace/training_trace/synthetic_train/* {TMP_TRACE_PATH}/",
        ]

        # **Run setup commands one by one, waiting for each to complete**
        for cmd in setup_commands:
            run_command_blocking(client, cmd, hostname)

        # After creating TMP_TRACE_PATH, copy traces
        LOCAL_TRACE_DIR = "/home/jane/Genet/abr_trace/testing_trace_mahimahi"
        # scp_to_remote(client, LOCAL_TRACE_DIR, TMP_TRACE_PATH, hostname)

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
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_BBA.sh {TMP_TRACE_PATH}/ 04_23_collect 6626 0 --collection",
            f"{ABR_SCRIPT_PATH}/run_mahimahi_emulation_MPC.sh {TMP_TRACE_PATH}/ 04_23_collect 6626 0 --collection",
        ]

        if node_id == 1:
            run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} '{abr_commands[0]}' C-m", hostname)
        else:
            run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} '{abr_commands[1]}' C-m", hostname)

        # **Execute ABR scripts in tmux, ensuring sequential execution**
        # for cmd in abr_commands:
        #     run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} '{cmd}' C-m", hostname)
        #     run_command_blocking(client, f"tmux send-keys -t {tmux_session_name} 'wait' C-m", hostname)

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
