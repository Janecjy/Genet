#!/usr/bin/env python3

import paramiko
import yaml
import concurrent.futures
import re

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"
USERNAME = "janechen"

# Read the server list from config.yaml
with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers"]

def get_last_epoch(client):
    """Retrieve the last logged epoch number from /mydata/logs/log_train."""
    log_file = "/mydata/logs/log_train"
    epoch_cmd = f"grep 'Epoch:' {log_file} | tail -1"

    try:
        stdin, stdout, stderr = client.exec_command(epoch_cmd)
        output = stdout.read().decode().strip()

        if not output:
            return None  # No epoch found

        # Extract epoch number using regex
        match = re.search(r"Epoch: (\d+)", output)
        if match:
            return int(match.group(1))

    except Exception as e:
        print(f"Error retrieving epoch: {e}")

    return None  # Default return if no epoch found

def check_logs_and_update(server_config):
    """SSH into the server, check the number of log files, get last epoch if applicable, and update config."""
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
            return (server, None, None)  # Return None if command failed

        log_count = int(output)
        print(f"[{server}] Log file count: {log_count}")

        last_epoch = None
        if log_count == 44:
            print(f"[{server}] Setting run: false in config.yaml")

            # Retrieve last epoch number from log_train
            last_epoch = get_last_epoch(client)
            if last_epoch is not None:
                print(f"[{server}] Last logged epoch: {last_epoch}")
                if last_epoch > 1:
                    print(f"[{server}] Setting run: false in config.yaml")
                    server_config["run"] = False
            else:
                print(f"[{server}] No epoch found in log_train")

        return (server, log_count, last_epoch)

    except Exception as e:
        print(f"Error checking logs on {server}: {e}")
        return (server, None, None)  # Return None if SSH failed
    finally:
        client.close()

if __name__ == "__main__":
    # Run checks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
        results = list(executor.map(check_logs_and_update, servers))

    # Update the config.yaml file
    with open(CONFIG_FILE, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print("Config file updated.")

    # Count how many servers have exactly 44 logs
    servers_with_44_logs = sum(1 for _, log_count, last_epoch in results if log_count == 44 and last_epoch and last_epoch > 1)
    total_servers = len(servers)
    servers_still_running = total_servers - servers_with_44_logs

    # Print summary
    if servers_with_44_logs == total_servers:
        print("✅ All servers have exactly 44 log files!")
    else:
        print(f"⚠️ {servers_still_running} servers are still running (less than 44 log files).")

    print("Log checks completed for all servers.")
