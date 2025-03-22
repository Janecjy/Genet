import paramiko
import yaml
import os
import argparse
import concurrent.futures

# Load test configuration from testconfig.yaml
CONFIG_FILE = "testconfig.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

# Separate nodes by branch type
unum_adaptor_nodes = [server for server in config["test_servers"] if server["branch"] == "unum-adaptor"]
sim_reproduce_nodes = [server for server in config["test_servers"] if server["branch"] == "sim-reproduce"]
other_nodes = [server for server in config["test_servers"] if server["branch"] not in ["unum-adaptor", "sim-reproduce"]]

# Parse command-line argument (only requires model directory name)
parser = argparse.ArgumentParser(description="Start remote test sessions on test nodes")
parser.add_argument("model_dir_name", help="Model directory name (e.g., 03_19_model_set)")
args = parser.parse_args()

username = "janechen"
GENET_BASE_PATH = "/users/janechen/Genet"
TEST_SCRIPT_PATH = f"{GENET_BASE_PATH}/src/emulator/abr/pensieve/drivers/run_models.sh"
SIM_SCRIPT_PATH = f"{GENET_BASE_PATH}/src/emulator/abr/pensieve/drivers/run_mahimahi_emulation_UDR_3.sh"
VIDEO_SERVER_PATH = f"{GENET_BASE_PATH}/src/emulator/abr/pensieve/video_server/video_server.py"

# Default argument values
MODEL_PATH = f"{GENET_BASE_PATH}/fig_reproduce/model/{args.model_dir_name}/"
TRACE_DIR = f"{GENET_BASE_PATH}/fig_reproduce/data/synthetic_test_mahimahi/"
SUMMARY_DIR = args.model_dir_name  # Same as model directory name
PORT_ID = "6626"
AGENT_ID = "0"
SEED = "0"
EXTRA_ARGS = "--use_embedding"
LOG_PATH = f"/mydata/logs/emu_test_{args.model_dir_name}.log"  # Log name matches model dir

# Total number of servers to assign
num_servers = 28  

# Adjust server distribution if unum-adaptor nodes are fewer than 28
num_nodes = len(unum_adaptor_nodes)
servers_per_node = max(1, num_servers // num_nodes) if num_nodes > 0 else 0
remaining_servers = num_servers % num_nodes  # Handle leftover servers

server_ranges = []
start_id = 1

# Assign server IDs only to `unum-adaptor` nodes
for i, server in enumerate(unum_adaptor_nodes):
    extra_server = 1 if i < remaining_servers else 0  # Distribute extra servers among first nodes
    end_id = start_id + servers_per_node + extra_server - 1
    server_ranges.append((server, start_id, end_id))
    start_id = end_id + 1

def start_remote_test(server, start_id=None, end_id=None):
    """SSH into a remote node, pull the latest code, start a tmux session, and run the test script."""
    node = server["hostname"]
    branch = server["branch"]
    redis_ip = server.get("redis_ip", "10.10.1.1")  # Default Redis IP if not specified

    print(f"Starting tests on {node}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(node, username=username)
        tmux_session_name = "main"
        test_window = "test_script"
        video_server_window = "video_server"

        # Commands to run before starting tmux
        commands = [
            "tmux kill-server || true",
            "rm -rf /mydata/logs/*",
            "mkdir -p /mydata/logs",
            f"cd {GENET_BASE_PATH} && git reset --hard && git fetch && git checkout {branch} && git pull",
            f"grep -rl --include='*.py' '10.10.1.2' {GENET_BASE_PATH}/src/emulator/abr/pensieve/ | xargs sed -i 's/10.10.1.2/{redis_ip}/g' || true",
            f"tmux new-session -d -s {tmux_session_name} 'bash'",

            # Start Video Server in a separate tmux window
            f"tmux new-window -t {tmux_session_name} -n {video_server_window}",
            f"tmux send-keys -t {tmux_session_name}:{video_server_window} 'source ~/miniconda/bin/activate genet_env' C-m",
            f"tmux send-keys -t {tmux_session_name}:{video_server_window} 'cd {GENET_BASE_PATH}/src/emulator/abr/pensieve/video_server' C-m",
            f"tmux send-keys -t {tmux_session_name}:{video_server_window} 'python video_server.py --port={PORT_ID}' C-m",

            # Start Test Script in a separate tmux window
            f"tmux new-window -t {tmux_session_name} -n {test_window}",
            f"tmux send-keys -t {tmux_session_name}:{test_window} 'source ~/miniconda/bin/activate genet_env' C-m",
            f"tmux send-keys -t {tmux_session_name}:{test_window} 'cd ~/Genet/src/emulator/abr' C-m",
        ]

        # Determine script execution
        if branch == "sim-reproduce":
            commands.append(
                f"tmux send-keys -t {tmux_session_name}:{test_window} 'bash {SIM_SCRIPT_PATH} ' C-m"
            )
        elif start_id and end_id:
            commands.append(
                f"tmux send-keys -t {tmux_session_name}:{test_window} "
                f"'bash {TEST_SCRIPT_PATH} {MODEL_PATH} {TRACE_DIR} {SUMMARY_DIR} {PORT_ID} {AGENT_ID} {EXTRA_ARGS} {SEED} {start_id} {end_id}' C-m"
            )

        commands.append("tmux ls")

        # Execute each command over SSH
        for cmd in commands:
            print("Executing command:", cmd)
            stdin, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode()
            err = stderr.read().decode()
            if out.strip():
                print(f"Output from {node}:\n{out}")
            if err.strip():
                print(f"Error from {node}:\n{err}")

        print(f"Tests started on {node}. Logs at {LOG_PATH}")

    except Exception as e:
        print(f"Error starting tests on {node}: {e}")

    finally:
        client.close()

# Run test setups in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=len(config["test_servers"])) as executor:
    futures = {}

    # Start test runs on `unum-adaptor` nodes (with assigned server IDs)
    for server, start_id, end_id in server_ranges:
        futures[executor.submit(start_remote_test, server, start_id, end_id)] = server["hostname"]

    # Start sim-reproduce script execution
    for server in sim_reproduce_nodes:
        futures[executor.submit(start_remote_test, server)] = server["hostname"]

    # Start git pull only for other nodes
    for server in other_nodes:
        futures[executor.submit(start_remote_test, server)] = server["hostname"]

    for future in concurrent.futures.as_completed(futures):
        node = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error on node {node}: {e}")

print("All remote tests have been started successfully.")
