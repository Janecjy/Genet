#!/usr/bin/env python3

import paramiko
import yaml
import os
import argparse
import concurrent.futures
import subprocess

def copy_traces_to_remote(hostname, username, local_trace_dir):
    """Copy trace files to the remote host."""
    remote_path = f"{username}@{hostname}:~/Genet/abr_trace/"
    
    print(f"[{hostname}] Copying traces from {local_trace_dir}...")
    try:
        subprocess.run(["scp", "-r", local_trace_dir, remote_path], check=True)
        print(f"[{hostname}] ✅ Traces copied successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{hostname}] ❌ Failed to copy traces: {e}")
        return False

def start_remote_test(server, model_path, trace_dir, summary_dir, port_id, agent_id, extra_args, username):
    """SSH into a remote node and run the test."""
    hostname = server["hostname"]
    branch = server["branch"]
    redis_ip = server.get("redis_ip", "10.10.1.1")

    print(f"\n=== Starting test on {hostname} (branch: {branch}) ===")

    # Copy traces to remote
    local_trace_dir = os.path.expanduser(trace_dir)
    if not copy_traces_to_remote(hostname, username, local_trace_dir):
        return False

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=username)
        
        # Remote paths
        genet_path = "/users/janechen/Genet"
        test_script_path = f"{genet_path}/scripts/additional_inputs/test_trained_models.sh"
        remote_trace_dir = f"{genet_path}/abr_trace/testing_trace_mahimahi_sample"
        log_path = f"/mydata/logs/test_{summary_dir}_{hostname.split('.')[0]}.log"

        # Commands to set up and run test
        commands = [
            "tmux kill-server || true",
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
            f"tmux send-keys -t test_main:test_script "
            f"'{test_script_path} {model_path} {remote_trace_dir} {summary_dir} {port_id} {agent_id} \"{extra_args}\" 2>&1 | tee {log_path}' C-m",
            
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

        print(f"[{hostname}] ✅ Test started successfully")
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
    parser.add_argument("--model-path", default="/mydata/results/abr/udr3_emu_par_emulation",
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
    parser.add_argument("--config-file", default="config.yaml",
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

    servers = config["servers"]
    username = "janechen"

    # Use all servers
    active_servers = servers

    # Limit number of servers if specified
    if args.max_servers:
        active_servers = active_servers[:args.max_servers]

    print(f"🎯 Test Configuration:")
    print(f"   Model path: {args.model_path}")
    print(f"   Trace dir: {args.trace_dir}")
    print(f"   Summary dir: {args.summary_dir}")
    print(f"   Port: {args.port_id}")
    print(f"   Agent ID: {args.agent_id}")
    print(f"   Extra args: {args.extra_args}")
    print(f"   Active servers: {len(active_servers)}")

    if args.dry_run:
        print("\n🔍 DRY RUN - Would run tests on:")
        for server in active_servers:
            print(f"   - {server['hostname']} (branch: {server['branch']})")
        return 0

    print(f"\n🚀 Starting tests on {len(active_servers)} servers...")

    # Run tests in parallel
    successful = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_servers)) as executor:
        futures = {
            executor.submit(
                start_remote_test, 
                server, 
                args.model_path, 
                args.trace_dir, 
                args.summary_dir, 
                args.port_id, 
                args.agent_id, 
                args.extra_args, 
                username
            ): server['hostname']
            for server in active_servers
        }

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
        print(f"\n🎉 Tests are running on {successful} servers!")
        print(f"💡 To monitor progress, SSH into a server and run:")
        print(f"   tmux attach -t test_main")
        print(f"   # Then switch between windows with Ctrl+b followed by 'n' or 'p'")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
