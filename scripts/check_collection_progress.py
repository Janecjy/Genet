import paramiko
import yaml
import argparse
import concurrent.futures

# Constants
USERNAME = "janechen"
TARGET_PATH = "/mydata/results/03_24_collect/*/*.out"

# Load node configuration
CONFIG_FILE = "config.yaml"
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

nodes = config["servers"]
NUM_NODES = len(nodes)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Check how many .out files exist on each node")
args = parser.parse_args()

def count_out_files_on_node(node):
    """SSH into the node and count the number of .out files."""
    hostname = node["hostname"]

    print(f"Checking .out files on {hostname}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=USERNAME)

        # Command to count the .out files
        cmd = f"ls {TARGET_PATH} 2>/dev/null | wc -l"

        stdin, stdout, stderr = client.exec_command(cmd)
        count = stdout.read().decode().strip()

        if not count:
            count = "0"

        print(f"[{hostname}] Found {count} .out files")

        return hostname, int(count)

    except Exception as e:
        print(f"[{hostname}] Error: {e}")
        return hostname, None

    finally:
        client.close()

# Run on all nodes in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_NODES) as executor:
    results = list(executor.map(count_out_files_on_node, nodes))

# Print summary
print("\n=== Summary of .out Files Per Node ===")
for hostname, count in results:
    if count is not None:
        print(f"{hostname}: {count} files")
    else:
        print(f"{hostname}: Error occurred")

print("\nCheck complete!")
