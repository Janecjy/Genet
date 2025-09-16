#!/usr/bin/env python3

import paramiko
import yaml
import os
import argparse
import concurrent.futures
import subprocess
import re
import math

def auto_detect_server_count_from_config():
    """
    Auto-detect number of server models from train.py configuration.
    Based on the training configuration logic.
    """
    # From train.py configuration (matching test_trained_models.sh)
    adaptor_inputs = ["original_selection"]  # Only this input type is used
    adaptor_hidden_layers = [128, 256]
    seeds = [10, 20, 30, 40, 50]
    context_windows = [1, 3, 5]
    
    # Calculate total configurations (same as train.py)
    base_combos = len(adaptor_inputs) * len(adaptor_hidden_layers) * len(context_windows)  # 1*2*3=6
    total_configs = base_combos * len(seeds)  # 6*5=30
    
    print(f"🔍 Auto-detected configuration:")
    print(f"   Input types: {len(adaptor_inputs)} ({adaptor_inputs})")
    print(f"   Hidden layers: {len(adaptor_hidden_layers)} ({adaptor_hidden_layers})")
    print(f"   Context windows: {len(context_windows)} ({context_windows})")
    print(f"   Seeds: {len(seeds)} ({seeds})")
    print(f"   Total server models expected: {total_configs}")
    
    return total_configs

def calculate_server_ranges_auto(active_nodes, total_servers_to_test=None, start_server_id=1):
    """
    Calculate server ID ranges for each test node.
    Auto-detects number of servers if not specified.
    
    Args:
        active_nodes: List of active test nodes
        total_servers_to_test: Total number of servers to test (auto-detect if None)
        start_server_id: Starting server ID (default: 1)
    
    Returns:
        List of tuples: (node, start_server_id, end_server_id) for each active node
    """
    if not active_nodes:
        return []
    
    # Auto-detect if not specified
    if total_servers_to_test is None:
        total_servers_to_test = auto_detect_server_count_from_config()
    
    num_nodes = len(active_nodes)
    servers_per_node = max(1, total_servers_to_test // num_nodes) if num_nodes > 0 else 0
    remaining_servers = total_servers_to_test % num_nodes  # Handle leftover servers
    
    print(f"📊 Distribution calculation:")
    print(f"   Total servers to test: {total_servers_to_test} (starting from server_{start_server_id})")
    print(f"   Active nodes: {num_nodes}")
    print(f"   Base servers per node: {servers_per_node}")
    print(f"   Extra servers for first {remaining_servers} nodes: 1 each")
    
    server_ranges = []
    current_start_id = start_server_id
    
    # Assign server IDs to nodes (exactly like emu_test.py)
    for i, node in enumerate(active_nodes):
        extra_server = 1 if i < remaining_servers else 0  # Distribute extra servers among first nodes
        servers_for_this_node = servers_per_node + extra_server
        
        if servers_for_this_node > 0:
            end_id = current_start_id + servers_for_this_node - 1
            server_ranges.append((node, current_start_id, end_id))
            current_start_id = end_id + 1
        else:
            server_ranges.append((node, None, None))  # No work for this node
    
    return server_ranges

def filter_nodes_by_branch(servers, target_branch=None):
    """Filter nodes by branch type, similar to emu_test.py."""
    if not target_branch:
        return servers
    
    filtered = [server for server in servers if server.get("branch") == target_branch]
    print(f"🔍 Filtered {len(filtered)} nodes with branch '{target_branch}' from {len(servers)} total nodes")
    return filtered

def discover_local_server_models(local_model_path):
    """
    Discover server model directories from local path.
    
    Args:
        local_model_path: Local path to model directory (e.g., /home/jane/Genet/fig_reproduce/model/09_04_model_set)
    
    Returns:
        List of server IDs found in the directory
    """
    if not os.path.exists(local_model_path):
        print(f"❌ Local model path does not exist: {local_model_path}")
        return []
    
    print(f"🔍 Discovering server models in: {local_model_path}")
    
    server_dirs = []
    try:
        for item in os.listdir(local_model_path):
            item_path = os.path.join(local_model_path, item)
            if os.path.isdir(item_path) and item.startswith('server_'):
                # Extract server number
                match = re.match(r'server_(\d+)', item)
                if match:
                    server_id = int(match.group(1))
                    server_dirs.append(server_id)
        
        server_dirs.sort()
        print(f"📊 Found {len(server_dirs)} server models: server_{min(server_dirs)} to server_{max(server_dirs)}")
        return server_dirs
        
    except Exception as e:
        print(f"❌ Error discovering local models: {e}")
        return []

def calculate_server_ranges_from_local(active_nodes, local_model_path):
    """
    Calculate server ID ranges for each test node based on local model discovery.
    
    Args:
        active_nodes: List of active test nodes  
        local_model_path: Local path to discover models from
    
    Returns:
        List of tuples: (node, start_server_id, end_server_id) for each active node
    """
    if not active_nodes:
        return []
    
    # Discover available server models locally
    server_ids = discover_local_server_models(local_model_path)
    if not server_ids:
        return []
    
    total_servers = len(server_ids)
    num_nodes = len(active_nodes)
    servers_per_node = max(1, total_servers // num_nodes) if num_nodes > 0 else 0
    remaining_servers = total_servers % num_nodes  # Handle leftover servers
    
    print(f"📊 Distribution calculation:")
    print(f"   Total servers to distribute: {total_servers}")
    print(f"   Active nodes: {num_nodes}")
    print(f"   Base servers per node: {servers_per_node}")
    print(f"   Extra servers for first {remaining_servers} nodes: 1 each")
    
    server_ranges = []
    current_idx = 0
    
    # Assign server IDs to nodes
    for i, node in enumerate(active_nodes):
        extra_server = 1 if i < remaining_servers else 0  # Distribute extra servers among first nodes
        servers_for_this_node = servers_per_node + extra_server
        
        if servers_for_this_node > 0 and current_idx < len(server_ids):
            start_server_id = server_ids[current_idx]
            end_idx = min(current_idx + servers_for_this_node - 1, len(server_ids) - 1)
            end_server_id = server_ids[end_idx]
            
            server_ranges.append((node, start_server_id, end_server_id))
            current_idx = end_idx + 1
        else:
            server_ranges.append((node, None, None))  # No work for this node
    
    return server_ranges

def copy_traces_to_remote(hostname, username, local_trace_dir):
    """Copy trace files to the remote host."""
    remote_path = f"{username}@{hostname}:~/Genet/abr_trace/"
    
    print(f"[{hostname}] Deleting existing traces on remote server...")
    try:
        # Delete existing traces on remote server
        subprocess.run(["ssh", f"{username}@{hostname}", "rm -rf ~/Genet/abr_trace/*"], check=True)
        print(f"[{hostname}] ✅ Existing traces deleted successfully")
    except subprocess.CalledProcessError as e:
        print(f"[{hostname}] ⚠️  Warning: Failed to delete existing traces: {e}")
    
    print(f"[{hostname}] Copying traces from {local_trace_dir}...")
    try:
        subprocess.run(["scp", "-r", local_trace_dir, remote_path], check=True)
        print(f"[{hostname}] ✅ Traces copied successfully")
        
        # Check how many files are on the remote server after copying
        result = subprocess.run(["ssh", f"{username}@{hostname}", "find ~/Genet/abr_trace -type f | wc -l"], 
                               capture_output=True, text=True, check=True)
        file_count = result.stdout.strip()
        print(f"[{hostname}] 📊 Total files on remote server after copy: {file_count}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{hostname}] ❌ Failed to copy traces: {e}")
        return False

def copy_server_models_to_remote(hostname, username, local_model_path, start_server_id, end_server_id):
    """Copy only the specified server models to the remote host."""
    if start_server_id is None or end_server_id is None:
        print(f"[{hostname}] No server range specified, skipping model copy")
        return True
    
    print(f"[{hostname}] Copying server models {start_server_id}-{end_server_id} from {local_model_path}...")
    
    # Create remote model directory
    remote_model_base = "/mydata/results/abr/udr3_emu_par_emulation"
    try:
        subprocess.run(["ssh", f"{username}@{hostname}", f"mkdir -p {remote_model_base}"], check=True)
        print(f"[{hostname}] ✅ Remote model directory created")
    except subprocess.CalledProcessError as e:
        print(f"[{hostname}] ❌ Failed to create remote model directory: {e}")
        return False
    
    # Copy only the specified server models
    models_copied = 0
    for server_id in range(start_server_id, end_server_id + 1):
        server_dir = f"server_{server_id}"
        local_server_path = os.path.join(local_model_path, server_dir)
        
        if os.path.exists(local_server_path):
            try:
                # Copy the entire server directory
                subprocess.run(["scp", "-r", local_server_path, f"{username}@{hostname}:{remote_model_base}/"], check=True)
                models_copied += 1
                print(f"[{hostname}] ✅ Copied {server_dir}")
            except subprocess.CalledProcessError as e:
                print(f"[{hostname}] ❌ Failed to copy {server_dir}: {e}")
                return False
        else:
            print(f"[{hostname}] ⚠️  Warning: Local server directory not found: {local_server_path}")
    
    print(f"[{hostname}] 📊 Successfully copied {models_copied} server models")
    
    # Verify models on remote server
    try:
        result = subprocess.run(["ssh", f"{username}@{hostname}", f"find {remote_model_base} -name 'server_*' -type d | wc -l"], 
                               capture_output=True, text=True, check=True)
        remote_model_count = result.stdout.strip()
        print(f"[{hostname}] 📊 Total server models on remote: {remote_model_count}")
    except subprocess.CalledProcessError as e:
        print(f"[{hostname}] ⚠️  Warning: Could not verify remote model count: {e}")
    
    return True

def start_remote_test(server, model_path, trace_dir, summary_dir, port_id, agent_id, extra_args, username, start_server_id=None, end_server_id=None, local_model_path=None):
    """SSH into a remote node and run the test."""
    hostname = server["hostname"]
    branch = server["branch"]
    redis_ip = server.get("redis_ip", "10.10.1.1")

    print(f"\n=== Starting test on {hostname} (branch: {branch}) ===")

    # Copy traces to remote
    local_trace_dir = os.path.expanduser(trace_dir)
    if not copy_traces_to_remote(hostname, username, local_trace_dir):
        return False

    # Copy only the corresponding server models to remote
    if local_model_path and start_server_id is not None and end_server_id is not None:
        if not copy_server_models_to_remote(hostname, username, local_model_path, start_server_id, end_server_id):
            return False
    else:
        print(f"[{hostname}] Using existing models on remote server at {model_path}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=username)
        
        # Remote paths
        genet_path = "/users/janechen/Genet"
        test_script_path = f"{genet_path}/scripts/additional_inputs/test_trained_models.sh"
        remote_trace_dir = f"{genet_path}/abr_trace/testing_trace_mahimahi_sample"
        log_path = f"/mydata/logs/test_{summary_dir}_{hostname.split('.')[0]}.log"

        # Build the test command with optional server range parameters
        test_cmd = f"{test_script_path} {model_path} {remote_trace_dir} {summary_dir} {port_id} {agent_id} \"{extra_args}\""
        if start_server_id is not None and end_server_id is not None:
            test_cmd += f" 42 {start_server_id} {end_server_id}"  # 42 is default seed
        
        # Commands to set up and run test
        commands = [
            "tmux kill-server || true",
            "mkdir -p /mydata/logs",
            f"cd {genet_path} && git reset --hard && git fetch && git checkout {branch} && git pull",
            f"grep -rl --include='*.py' '10.10.1.2' {genet_path}/src/emulator/abr/pensieve/ | xargs sed -i 's/10.10.1.2/{redis_ip}/g' || true",
            "tmux new-session -d -s test_main 'bash'",
            
            # Start video server in separate window
            "tmux new-window -t test_main -n video_server",
            f"tmux send-keys -t test_main:video_server 'source ~/miniconda/bin/activate genet_env' C-m",
            f"tmux send-keys -t test_main:video_server 'cd {genet_path}/src/emulator/abr/pensieve/video_server' C-m",
            f"tmux send-keys -t test_main:video_server 'python video_server.py --port={port_id}' C-m",
            
            # Start test script in separate window
            "tmux new-window -t test_main -n test_script",
            f"tmux send-keys -t test_main:test_script 'source ~/miniconda/bin/activate genet_env' C-m",
            f"tmux send-keys -t test_main:test_script 'cd {genet_path}/scripts/additional_inputs' C-m",
            f"tmux send-keys -t test_main:test_script '{test_cmd} 2>&1 | tee {log_path}' C-m",
            
            "tmux ls"
        ]

        # Execute commands
        for cmd in commands:
            print(f"[{hostname}] Executing: {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            
            if out:
                print(f"[{hostname}] Output: {out}")
            if err:
                print(f"[{hostname}] Error: {err}")

        range_info = ""
        if start_server_id is not None and end_server_id is not None:
            range_info = f" (testing servers {start_server_id}-{end_server_id})"
        
        print(f"[{hostname}] ✅ Test started successfully{range_info}")
        print(f"[{hostname}] 📁 Results will be in: /mydata/results/")
        print(f"[{hostname}] 📝 Logs at: {log_path}")
        return True

    except Exception as e:
        print(f"[{hostname}] ❌ Error: {e}")
        return False
    finally:
        client.close()

def main():
    parser = argparse.ArgumentParser(description="Run trained model tests on remote servers")
    parser.add_argument("local_model_path", 
                       help="Local path to model directory (e.g., /home/jane/Genet/fig_reproduce/model/09_04_model_set)")
    parser.add_argument("--remote-model-path", default="/mydata/results/abr/udr3_emu_par_emulation",
                       help="Path to trained models directory on remote servers")
    parser.add_argument("--trace-dir", default="~/Genet/abr_trace/testing_trace_mahimahi_sample",
                       help="Local trace directory to copy to servers")
    parser.add_argument("--summary-dir", default="test_results",
                       help="Summary directory name for results")
    parser.add_argument("--port-id", default="6626",
                       help="Port ID for testing")
    parser.add_argument("--agent-id", default="0", 
                       help="Agent ID for testing")
    parser.add_argument("--extra-args", default="--use_embedding",
                       help="Extra arguments for testing")
    parser.add_argument("--config-file", default="testconfig.yaml",
                       help="Server configuration file")
    parser.add_argument("--max-servers", type=int, default=None,
                       help="Maximum number of servers to use (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    args = parser.parse_args()

    # Load server configuration
    if not os.path.exists(args.config_file):
        print(f"Error: Configuration file '{args.config_file}' not found!")
        return 1

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    servers = config["test_servers"]  # Use test_servers instead of servers
    username = "janechen"

    # Use all servers (all are on raw-add-inputs branch)
    active_servers = servers

    # Limit number of servers if specified
    if args.max_servers:
        active_servers = active_servers[:args.max_servers]

    print(f"🎯 Test Configuration:")
    print(f"   Local model path: {args.local_model_path}")
    print(f"   Remote model path: {args.remote_model_path}")
    print(f"   Trace dir: {args.trace_dir}")
    print(f"   Summary dir: {args.summary_dir}")
    print(f"   Port: {args.port_id}")
    print(f"   Agent ID: {args.agent_id}")
    print(f"   Extra args: {args.extra_args}")
    print(f"   Active servers: {len(active_servers)}")

    # Always calculate distribution based on local model discovery
    if not active_servers:
        print("❌ No active servers available for distribution")
        return 1
        
    # Discover and distribute server models from local path
    server_ranges = calculate_server_ranges_from_local(active_servers, args.local_model_path)
    
    if not server_ranges:
        print("❌ No server models found for distribution")
        return 1
        
    print(f"\n📊 Model Distribution Plan:")
    for server, start_id, end_id in server_ranges:
        if start_id is not None and end_id is not None:
            model_count = end_id - start_id + 1
            print(f"   {server['hostname']}: servers {start_id}-{end_id} ({model_count} models)")
        else:
            print(f"   {server['hostname']}: no models assigned")

    if args.dry_run:
        print("\n🔍 DRY RUN - Would run tests on:")
        for server, start_id, end_id in server_ranges:
            if start_id is not None and end_id is not None:
                print(f"   - {server['hostname']} (branch: {server['branch']}) - servers {start_id}-{end_id}")
            else:
                print(f"   - {server['hostname']} (branch: {server['branch']}) - no models assigned")
        return 0

    print(f"\n🚀 Starting tests on {len(active_servers)} servers...")

    # Run tests in parallel
    successful = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_servers)) as executor:
        futures = {}
        
        # Always use distribution from server_ranges
        for server, start_id, end_id in server_ranges:
            if start_id is not None and end_id is not None:
                future = executor.submit(
                    start_remote_test, 
                    server, 
                    args.remote_model_path, 
                    args.trace_dir, 
                    args.summary_dir, 
                    args.port_id, 
                    args.agent_id, 
                    args.extra_args, 
                    username,
                    start_id,
                    end_id,
                    args.local_model_path
                )
                futures[future] = server['hostname']
            else:
                print(f"[{server['hostname']}] Skipping - no models assigned in distribution")

        for future in concurrent.futures.as_completed(futures):
            hostname = futures[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[{hostname}] ❌ Exception: {e}")
                failed += 1

    # Summary
    print(f"\n=== Test Execution Summary ===")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(active_servers)}")
    
    if successful > 0:
        print(f"\n🎉 Tests are running on {successful} servers with distributed model ranges!")
        print(f"💡 To monitor progress, SSH into a server and run:")
        print(f"   tmux attach -t test_main")
        print(f"   # Then switch between windows with Ctrl+b followed by 'n' or 'p'")
        
        print(f"\n📊 Model distribution summary:")
        total_models_assigned = 0
        for server, start_id, end_id in server_ranges:
            if start_id is not None and end_id is not None:
                model_count = end_id - start_id + 1
                total_models_assigned += model_count
                print(f"   {server['hostname']}: {model_count} models")
        print(f"   Total models being tested: {total_models_assigned}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
