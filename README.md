# Unum Integration with Pensieve (ABR)

This repository extends the [Pensieve](https://github.com/hongzimao/pensieve) implementation from the [Genet](https://github.com/GenetProject/Genet) codebase to integrate **Unum adaptor** with adaptive bitrate (ABR) video streaming.

Here is a video step-by-step tutorial demonstrating how to set up a simple test for the artifact evaluation: https://drive.google.com/file/d/1dXcBlECiazWIUwPtp6NtRrc1FZrDwzZ8/view?usp=share_link

## Overview

**Genet with Unum** provides a reinforcement learning framework for adaptive bitrate streaming that combines:
- **Pensieve baseline**: Original RL-based ABR algorithm using A3C
- **Unum adaptor**: An enhancement layer that adapts the original policy to new network conditions with minimal retraining

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Artifact Evaluation Quick Start](#artifact-evaluation-quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## Prerequisites

### System Requirements

- **OS**: Ubuntu 24.04 (tested on CloudLab)
- **Storage**: `/mydata` mount point for temporary files (configurable)

### Dependencies

The setup script automatically installs:
- TensorFlow 1.15 (with GPU support)
- PyTorch
- Mahimahi network emulator
- Selenium for browser emulation
- Redis for distributed coordination

---

## Artifact Evaluation Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Janecjy/Genet.git
cd Genet
```

### 2. Configure Your Nodes

Edit the configuration files with your server details:

**For testing** (`scripts/testconfig.yaml`):
```yaml
username: your_username
test_servers:
  - hostname: test-node1.cloudlab.us
    branch: sim-reproduce  # Original Pensieve
    redis_ip: 10.10.1.1
    redis: false
  - hostname: test-node2.cloudlab.us
    branch: main          # Unum adaptor
    redis_ip: 10.10.1.2
    redis: true
```

### 3. Automated Setup

From the `scripts/` directory:

```bash
# Set up testing nodes
python emu_setup.py --mode test
```

This will:
- Install all dependencies on remote nodes
- Configure the network environment
- Set up Redis servers for distributed coordination

### 4. Run Tests

**Single-server mode** (Artifact Evaluation):
```bash
cd scripts/
python emu_test.py unum_adaptor --single-server
```

### 5. View Results

Results are saved in:
- **Original Pensieve**: `results/pensieve/UDR-3_60_40/`
- **Unum Adaptor**: `/mydata/results/unum_adaptor/unum_adaptor/UDR-3_0_60_40/`

---

## Configuration

### Training Configuration (`config.yaml`)

```yaml
username: your_ssh_username

servers:
  - branch: unum-adaptor           # Branch to use
    hostname: node.cloudlab.us     # Server hostname
    redis: true                    # Enable Redis coordination
    redis_ip: 10.10.1.1           # Redis server IP
    run: true                      # Include in training run
    scp_extra_path: /path/traces  # Optional: additional trace paths
```

**Configuration parameters**:
- `branch`: `unum-adaptor` for Unum training, `sim-reproduce` for baseline
- `redis`: Enable Redis for multi-node coordination. `sim-reproduce` doesn't require Redis.
- `redis_ip`: Unique IP for each Redis instance
- `run`: Set to `false` to skip this server during training
- `scp_extra_path`: Optional path to copy additional network traces

### Testing Configuration (`testconfig.yaml`)

Similar structure to training config, but uses `test_servers` key:

```yaml
username: your_ssh_username

test_servers:
  - hostname: test1.cloudlab.us
    branch: sim-reproduce    # Original Pensieve baseline
    redis_ip: 10.10.1.1
    redis: false
  
  - hostname: test2.cloudlab.us
    branch: main            # Unum adaptor
    redis_ip: 10.10.1.2
    redis: true
```

### Adaptor Hyperparameters

Edit `scripts/emu_train_embedding.py` or `scripts/additional_inputs/train.py`:

```python
# Input types for the adaptor
adaptor_inputs = ["original_selection", "hidden_state"]

# Hidden layer sizes
adaptor_hidden_layers = [128, 256]

# Random seeds for reproducibility
seeds = [10, 20, 30, 40, 50]
```

---

## Project Structure

```
Genet/
├── README.md                       # This file
├── scripts/                        # Automation scripts
│   ├── config.yaml                # Training node config
│   ├── testconfig.yaml            # Testing node config
│   ├── emu_setup.py               # Automated setup
│   ├── emu_test.py                # Testing orchestration
│   ├── emu_train_embedding.py     # Training orchestration
│   ├── plot_*.py                  # Result visualization
│   └── additional_inputs/         # Extended configurations
│
├── src/                           # Source code
│   ├── emulator/                  # ABR emulation
│   │   └── abr/
│   │       ├── pensieve/          # Unum-enhanced Pensieve
│   │       │   ├── a3c/           # A3C implementation
│   │       │   │   ├── adaptor.py # Unum adaptor network
│   │       │   │   ├── network.py # Base networks
│   │       │   │   └── a3c_jump.py
│   │       │   ├── agent_policy/  # Policy implementations
│   │       │   ├── drivers/       # Emulation drivers
│   │       │   └── virtual_browser/
│   │       ├── pensieve_orig/     # Original Pensieve
│   ├── simulator/                 # Simulation environment
│   └── common/                    # Shared utilities
│
├── abr_trace/                     # Network traces
│
├── config/                        # Emulation parameters
```