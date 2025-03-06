import paramiko
import time
import os
import yaml
import concurrent.futures
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Setup Genet on remote servers")
args = parser.parse_args()

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

servers = config["servers"]
username = "janechen"

# Paths for local and remote chromedriver
LOCAL_CHROMEDRIVER_PATH = "/home/jane/Downloads/chromedriver"
REMOTE_CHROMEDRIVER_PATH = "/users/janechen/Genet/src/emulator/abr/pensieve/virtual_browser/abr_browser_dir/"

# Set custom Redis port
REDIS_PORT = 2666  # Change this if needed

def scp_file(server):
    """ SCP the chromedriver file to the remote server """
    print(f"Transferring chromedriver to {server}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(server, username=username)
        sftp = client.open_sftp()

        # Ensure the remote directory exists
        try:
            sftp.stat(REMOTE_CHROMEDRIVER_PATH)
        except FileNotFoundError:
            sftp.mkdir(REMOTE_CHROMEDRIVER_PATH)

        sftp.put(LOCAL_CHROMEDRIVER_PATH, os.path.join(REMOTE_CHROMEDRIVER_PATH, os.path.basename(LOCAL_CHROMEDRIVER_PATH)))
        print(f"Chromedriver successfully transferred to {server}:{REMOTE_CHROMEDRIVER_PATH}")

        sftp.close()
    except Exception as e:
        print(f"Error transferring chromedriver to {server}: {e}")
    finally:
        client.close()

def run_remote_commands(server, commands):
    """ SSH into the server and execute the given commands """
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
    """ Run full setup process for a single server in parallel """
    server = server_config["hostname"]
    branch = server_config["branch"]
    redis_node = server_config["redis"]

    # Define Redis IP based on server index (starting from 1)
    redis_ip = f"10.10.1.{server_index + 1}"

    setup_commands = [
        "sudo DEBIAN_FRONTEND=noninteractive apt-get update -y",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git apt-transport-https ca-certificates curl software-properties-common tmux xvfb",
        "sudo chown -R janechen /mydata/",
        
        # Install Mahimahi
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential git debhelper autotools-dev dh-autoreconf iptables protobuf-compiler libprotobuf-dev pkg-config libssl-dev dnsmasq-base ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev iproute2 apache2-dev apache2-bin iptables dnsmasq-base gnuplot iproute2 apache2-api-20120211 libwww-perl",
        "[ -d ~/mahimahi ] || git clone https://github.com/ravinet/mahimahi ~/mahimahi",
        "cd ~/mahimahi && ./autogen.sh && ./configure && make && sudo make install",

        # # Install BPFTrace
        # "echo 'deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse' | sudo tee -a /etc/apt/sources.list.d/ddebs.list",
        # "echo 'deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse' | sudo tee -a /etc/apt/sources.list.d/ddebs.list",
        # "echo 'deb http://ddebs.ubuntu.com $(lsb_release -cs)-proposed main restricted universe multiverse' | sudo tee -a /etc/apt/sources.list.d/ddebs.list",
        # "sudo apt install -y ubuntu-dbgsym-keyring",
        # "sudo apt update",
        # "sudo apt install -y bpftrace-dbgsym",
        

        # Install Redis
        "sudo DEBIAN_FRONTEND=noninteractive apt install -y redis-server",
        
        # Install Miniconda and create Python 3.6 environment
        "[ -d ~/miniconda ] || (wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b -p ~/miniconda)",
        
        # Ensure Conda is initialized correctly
        "eval \"$(~/miniconda/bin/conda shell.bash hook)\" && echo $PATH && conda env list | grep genet_env || conda create -y -n genet_env python=3.6",
        
        # Activate Conda environment
        "source ~/miniconda/bin/activate genet_env",

        # Clone Genet repo and checkout correct branch
        "[ -d ~/Genet ] || git clone https://github.com/Janecjy/Genet.git ~/Genet",
        f"cd ~/Genet && git fetch --all && git checkout {branch} && git pull",
        "git config --global user.email janechen@cs.utexas.edu",
        "git config --global user.name Janecjy",

        # Install emulation dependencies
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -y install xvfb python3-pip python3-tk unzip libnss3",
        # "wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb",
        # "sudo DEBIAN_FRONTEND=noninteractive apt-get -yf install ./google-chrome-stable_current_amd64.deb",

        # Install Python dependencies inside Conda
        "source ~/miniconda/bin/activate genet_env && pip install numpy tensorflow==1.15.0 selenium pyvirtualdisplay numba torch tflearn xvfbwrapper matplotlib redis scipy pandas"
    ]
    
    # Redis configuration using the dynamically assigned IP
    if redis_node:
        redis_commands = [
            f"tmux new-session -d -s redis 'redis-server --port {REDIS_PORT} --bind {redis_ip} --protected-mode no'",
            f"echo 'Redis started on {redis_ip}:{REDIS_PORT} in tmux session on {server}'"
        ]
        setup_commands.extend(redis_commands)

    # Run setup commands
    run_remote_commands(server, setup_commands)

    # Transfer chromedriver
    scp_file(server)

    print(f"Setup completed for {server}.")

# Run setup on all servers in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as executor:
    futures = {executor.submit(setup_server, server_config, i): server_config["hostname"] for i, server_config in enumerate(servers)}
    
    for future in concurrent.futures.as_completed(futures):
        server = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error setting up {server}: {e}")

print("Setup completed on all servers.")
