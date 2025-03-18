#!/usr/bin/env python3

import paramiko
import yaml
import concurrent.futures

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"
USERNAME = "janechen"

# Read the server list from config.yaml
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers"]

def check_logs(server_config):
    """SSH into the server and check the number of log files in /mydata/logs."""
    server = server_config["hostname"]
    log_count_cmd = "ls /mydata/logs | wc -l"

    print(f"Checking logs on {server}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=USERNAME)
        stdin, stdout, stderr = client.exec_command(log_count_cmd)

        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()

        if error:
            print(f"[{server}] Error: {error}")
        else:
            print(f"[{server}] Log file count: {output}")

    except Exception as e:
        print(f"Error checking logs on {server}: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    # Run checks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
        executor.map(check_logs, servers)

    print("Log file counts retrieved for all servers.")
