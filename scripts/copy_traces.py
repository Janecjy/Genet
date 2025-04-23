import paramiko
import yaml
import argparse
import os
import concurrent.futures

# Constants
USERNAME = "janechen"
REMOTE_BASE_PATH = "/mydata/results/04_07_collect/"
LOCAL_BASE_PATH = "/home/jane/Genet/data/abr/unum/"
CONFIG_FILE = "config.yaml"

# Load node configuration
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

nodes = config["servers"]
NUM_NODES = len(nodes)

# Ensure base local directory exists
os.makedirs(LOCAL_BASE_PATH, exist_ok=True)

def scp_out_files(node):
    """SCP .out files from the remote node to local directory, preserving subdirectories."""
    hostname = node["hostname"]
    print(f"Transferring .out files from {hostname}...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, username=USERNAME)
        sftp = ssh.open_sftp()

        # Get list of subdirectories (BBA_0_60_40, RobustMPC_0_60_40)
        stdin, stdout, stderr = ssh.exec_command(f"ls -d {REMOTE_BASE_PATH}*/ 2>/dev/null")
        subdirs = stdout.read().decode().strip().split("\n")
        subdirs = [d.strip() for d in subdirs if d.strip()]

        total_files_transferred = 0
        
        for subdir in subdirs:
            subdir_name = os.path.basename(os.path.normpath(subdir))
            local_subdir = os.path.join(LOCAL_BASE_PATH, subdir_name)
            os.makedirs(local_subdir, exist_ok=True)
            
            # Get list of .out files in subdir
            stdin, stdout, stderr = ssh.exec_command(f"ls {subdir}/*.out 2>/dev/null")
            files = stdout.read().decode().strip().split("\n")
            files = [f.strip() for f in files if f.strip()]

            if not files:
                print(f"[{hostname}] No .out files found in {subdir}.")
                continue

            # Transfer files
            for remote_file in files:
                filename = os.path.basename(remote_file)
                local_file = os.path.join(local_subdir, filename)
                sftp.get(remote_file, local_file)

            print(f"[{hostname}] Transferred {len(files)} files from {subdir}.")
            total_files_transferred += len(files)

        return hostname, total_files_transferred

    except Exception as e:
        print(f"[{hostname}] Error: {e}")
        return hostname, None
    
    finally:
        ssh.close()

# Run SCP in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
    results = list(executor.map(scp_out_files, nodes))

# Print summary
print("\n=== Summary of Transferred .out Files Per Node ===")
for hostname, count in results:
    if count is not None:
        print(f"{hostname}: {count} files transferred")
    else:
        print(f"{hostname}: Error occurred")

print("\nTransfer complete!")
