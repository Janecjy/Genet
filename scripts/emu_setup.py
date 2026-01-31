import paramiko
import time
import os
import yaml
import concurrent.futures
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Setup Genet on remote servers")
parser.add_argument("--mode", choices=["train", "test"],
                    required=True, help="Specify 'train' or 'test' mode")
args = parser.parse_args()

# Select configuration based on mode
if args.mode == "train":
    CONFIG_FILE = "config.yaml"
elif args.mode == "test":
    CONFIG_FILE = "testconfig.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers" if args.mode == "train" else "test_servers"]
# Read username from config
username = config.get("username")
scp_extra_path = config.get("scp_extra_path")
REDIS_PORT = 2666  # Change if needed


def scp_files(server, scp_extra_path):
    """SCP files to the remote server (chromedriver + test traces if needed)"""
    print(f"Transferring files to {server}...")
    remote_data_path = f"/users/{username}/Genet/abr_trace/"
    os.system(
        f"scp -r {scp_extra_path} {username}@{server}:{remote_data_path}")
    print(f"Mahimahi traces successfully transferred to {server}")


def run_remote_commands(server, commands):
    """SSH into the server and execute the given commands"""
    print(f"Connecting to {server}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=username)
        for cmd in commands:
            print(f"Running on {server}: {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            time.sleep(2)
            print(stdout.read().decode())
            print(stderr.read().decode())

    finally:
        client.close()


def setup_server(server_config, server_index):
    """Run setup process for a single server in parallel"""
    server = server_config["hostname"]
    branch = server_config["branch"]
    redis_node = server_config["redis"]
    redis_ip = server_config["redis_ip"]

    setup_commands = [
        "sudo DEBIAN_FRONTEND=noninteractive apt-get update -y",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git apt-transport-https ca-certificates curl software-properties-common tmux xvfb",
        f"sudo chown -R {username} /mydata/",

        # Install Mahimahi
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential git debhelper autotools-dev dh-autoreconf iptables protobuf-compiler libprotobuf-dev pkg-config libssl-dev dnsmasq-base ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev iproute2 apache2-dev apache2-bin iptables dnsmasq-base gnuplot iproute2 apache2-api-20120211 libwww-perl",
        "[ -d ~/mahimahi ] || git clone https://github.com/ravinet/mahimahi ~/mahimahi",
        "cd ~/mahimahi && ./autogen.sh && ./configure && make && sudo make install",

        # Install BPFTrace
        "sudo sh -c 'echo \"deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse\" > /etc/apt/sources.list.d/ddebs.list'",
        "sudo sh -c 'echo \"deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse\" >> /etc/apt/sources.list.d/ddebs.list'",
        "sudo sh -c 'echo \"deb http://ddebs.ubuntu.com $(lsb_release -cs)-proposed main restricted universe multiverse\" >> /etc/apt/sources.list.d/ddebs.list'",

        # Install debug keyring and update package lists
        "sudo DEBIAN_FRONTEND=noninteractive apt install -y ubuntu-dbgsym-keyring",
        "sudo DEBIAN_FRONTEND=noninteractive apt update",

        # Install bpftrace debug symbols
        "sudo DEBIAN_FRONTEND=noninteractive apt install -y bpftrace-dbgsym",

        # Install Redis
        "sudo DEBIAN_FRONTEND=noninteractive apt install -y redis-server",

        # Install Miniconda and create Python 3.6 environment
        "[ -d ~/miniconda ] || (wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p ~/miniconda)",

        # Ensure Conda env exists (use conda-forge to avoid ToS acceptance requirement)
        "~/miniconda/bin/conda env list | grep -q \"genet_env\" || ~/miniconda/bin/conda create -y -n genet_env python=3.6 -c conda-forge --override-channels",

        # Clone Genet repo and checkout correct branch
        "[ -d ~/Genet ] || git clone https://github.com/Janecjy/Genet.git ~/Genet",
        f"cd ~/Genet && git fetch --all && git checkout main && git pull",

        # Install emulation dependencies
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -y install xvfb python3-pip python3-tk unzip",
        "wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -yf install ./google-chrome-stable_current_amd64.deb",

        # Install Python dependencies inside Conda using the env's pip (avoids activation persistence issues)
        "~/miniconda/envs/genet_env/bin/pip install numpy tensorflow==1.15.0 selenium pyvirtualdisplay numba torch tflearn xvfbwrapper matplotlib redis scipy pandas || true"

    ]

    # Redis configuration using the dynamically assigned IP
    if redis_node:
        redis_commands = [
            f"tmux new-session -d -s redis 'redis-server --port {REDIS_PORT} --bind {redis_ip} --protected-mode no'",
            f"echo 'Redis started on {redis_ip}:{REDIS_PORT} on {server}'"
        ]
        setup_commands.extend(redis_commands)

    # Run setup commands
    run_remote_commands(server, setup_commands)
    if scp_extra_path:
        scp_files(server, scp_extra_path)

    print(f"Setup completed for {server}.")


# Run setup on all servers in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
    futures = {executor.submit(setup_server, server_config, i)
                               : server_config["hostname"] for i, server_config in enumerate(servers)}

    for future in concurrent.futures.as_completed(futures):
        server = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error setting up {server}: {e}")

print(
    f"Setup completed on all {'training' if args.mode == 'train' else 'testing'} servers.")
