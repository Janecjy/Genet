# Unum Integration with Pensieve (ABR)

This repository extends the Pensieve implementation in the [Genet](https://github.com/GenetProject/Genet) codebase to integrate Unum with adaptive bitrate (ABR) selection.

Our integration adds support for training and evaluating the Unum adaptor, while preserving the original Pensieve baseline for comparison.

---

## Setup

We provide a setup script that installs all dependencies required for Unum adaptor training and testing on fresh Ubuntu 24.04 CloudLab nodes, assuming there is temporary filesystem mount point `/mydata`.

- **UNUM adaptor (training and testing)**: `main`
- **Original Pensieve baseline branch**: `sim-reproduce`

---

### Configuration

#### Training Nodes

List all training nodes in `config.yaml`.  
Each node should include:
- Its IP address (used as the Redis IP)
- Any additional optional configuration fields

#### Testing Nodes

List all testing nodes in `testconfig.yaml`, using the same format as the training configuration.

#### Optional Fields

- `scp_extra_path`: additional trace directories to copy to the node for training or testing

Refer to the example configuration files for the expected format.

---

### Setup Commands

From the scripts folder:

#### Set up training environment
```bash
python emu_setup.py --mode train
```

#### Set up testing environment
```bash
python emu_setup.py --mode test
```

## Test

Run tests on configured test nodes:

### Single-server mode (recommended for quick testing)
Tests one model on the first available unum-adaptor node:
```bash
python emu_test.py {model_name (e.g. unum_adaptor)} --single-server
```

### Multi-server mode (distributed testing)
Distributes multi-model testing across multiple servers:
```bash
python emu_test.py {model_name (e.g. unum_adaptor)}
```