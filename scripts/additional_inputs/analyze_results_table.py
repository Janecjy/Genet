#!/usr/bin/env python3
"""
Analyze test results and create a table of results based on context window and hidden size dimensions.
For each configuration, finds the best performing models from all configurations that belong to this setting.
"""

import os
import re
from collections import defaultdict

# Try to import optional packages, fall back to basic functionality if not available
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS_NUMPY = True
except ImportError:
    HAS_PANDAS_NUMPY = False
    print("Warning: pandas/numpy not available, using basic statistics")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Basic statistics functions for when numpy is not available
def mean(values):
    """Calculate mean of a list of values."""
    if HAS_PANDAS_NUMPY:
        return np.mean(values)
    return sum(values) / len(values) if values else 0

def std(values):
    """Calculate standard deviation of a list of values."""
    if HAS_PANDAS_NUMPY:
        return np.std(values)
    if not values:
        return 0
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5

def read_csv_whitespace(file_path):
    """Basic CSV reader for whitespace-delimited files."""
    if HAS_PANDAS_NUMPY:
        return pd.read_csv(file_path, delim_whitespace=True)
    
    # Basic fallback implementation
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None
        
        # Parse header
        headers = lines[0].strip().split()
        for header in headers:
            data[header] = []
        
        # Parse data
        for line in lines[1:]:
            values = line.strip().split()
            for i, header in enumerate(headers):
                if i < len(values):
                    try:
                        # Try to convert to float
                        data[header].append(float(values[i]))
                    except ValueError:
                        # Keep as string if not a number
                        data[header].append(values[i])
    
    return data

# ------------------ Directories ---------------------
RESULTS_BASE_DIR = "/home/jane/Genet/scripts/additional_inputs/09_16_model_set"
TEST_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "raw-add-inputs-test_results/test_results")
SAMPLE_DIR = "/home/jane/Genet/abr_trace/testing_trace_mahimahi_sample"

# Pensieve baseline directories (from plot_reward_boxplot.py)
OLD_RESULTS_DIR = "/home/jane/Genet/scripts/04_20_model_set"
PENSIEVE_DIR = os.path.join(OLD_RESULTS_DIR, "pensieve-original/testing_trace_mahimahi")

# ------------------ Configuration Parsing ---------------------
def parse_model_config(model_dir_name):
    """
    Parse model configuration from directory name like:
    server_1_original_selection_h512_cw1_s10_ep_2710_seed_10_adaptor_original_selection_hidden_512_cw_1
    
    Returns: dict with parsed configuration
    """
    # Extract key parameters using regex
    patterns = {
        'server_id': r'server_(\d+)',
        'hidden_size_old': r'_h(\d+)_',  # This might be incorrect/legacy
        'context_window': r'_cw(\d+)_',
        'seed': r'_s(\d+)_',
        'epoch': r'_ep_(\d+)_',
        'adaptor_hidden': r'hidden_(\d+)_cw',  # This is the actual trained config
        'adaptor_cw': r'cw_(\d+)$'
    }
    
    config = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, model_dir_name)
        if match:
            config[key] = int(match.group(1))
        else:
            config[key] = None
    
    # Use the actual trained hidden size (from checkpoint filename in directory name)
    config['hidden_size'] = config['adaptor_hidden']
    
    # Extract input type
    if 'original_selection' in model_dir_name:
        config['input_type'] = 'original_selection'
    elif 'hidden_state' in model_dir_name:
        config['input_type'] = 'hidden_state'
    else:
        config['input_type'] = 'unknown'
    
    return config

def is_valid_log(log_path):
    """Check if log file is valid and has reward data."""
    try:
        data = read_csv_whitespace(log_path)
        if HAS_PANDAS_NUMPY:
            return 'reward' in data.columns and len(data) > 2
        else:
            return 'reward' in data and len(data['reward']) > 2
    except:
        return False

def compute_mean_reward(log_path):
    """Compute mean reward from log file (excluding first row)."""
    try:
        data = read_csv_whitespace(log_path)
        if HAS_PANDAS_NUMPY:
            if 'reward' in data.columns and len(data) > 1:
                return data['reward'][1:].mean()
        else:
            if 'reward' in data and len(data['reward']) > 1:
                return mean(data['reward'][1:])
    except:
        return None
    return None

def get_sample_traces():
    """Get list of sample trace files."""
    if not os.path.exists(SAMPLE_DIR):
        print(f"Warning: Sample directory not found: {SAMPLE_DIR}")
        return []
    
    sample_traces = []
    for trace_type in ['fcc', 'norway']:
        traces = [f for f in os.listdir(SAMPLE_DIR) if f.startswith(f"{trace_type}-test")]
        sample_traces.extend([f"log_RL_{trace}" for trace in traces])
    
    return sample_traces

def get_pensieve_baseline_rewards(sample_traces):
    """Get Pensieve baseline rewards for sample traces."""
    pensieve_rewards = {}
    
    if not os.path.exists(PENSIEVE_DIR):
        print(f"Warning: Pensieve directory not found: {PENSIEVE_DIR}")
        return pensieve_rewards
    
    for trace_file in sample_traces:
        pensieve_path = os.path.join(PENSIEVE_DIR, trace_file)
        if os.path.exists(pensieve_path) and is_valid_log(pensieve_path):
            reward = compute_mean_reward(pensieve_path)
            if reward is not None:
                pensieve_rewards[trace_file] = reward
    
    return pensieve_rewards

def collect_model_results():
    """Collect results from all model directories."""
    models_data = {}
    sample_traces = get_sample_traces()
    pensieve_baseline = get_pensieve_baseline_rewards(sample_traces)
    
    if not sample_traces:
        print("No sample traces found, using all available traces")
        use_sample_filter = False
    else:
        print(f"Found {len(sample_traces)} sample traces to filter by")
        print(f"Found {len(pensieve_baseline)} Pensieve baseline rewards")
        use_sample_filter = True
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(TEST_RESULTS_DIR) if d.startswith("server_")]
    
    for model_dir in model_dirs:
        model_path = os.path.join(TEST_RESULTS_DIR, model_dir)
        config = parse_model_config(model_dir)
        
        # Skip if we couldn't parse the config properly
        if config['hidden_size'] is None or config['context_window'] is None:
            print(f"Warning: Could not parse config for {model_dir}")
            continue
        
        # Look for log files in UDR-3_0_60_40 subdirectory
        logs_dir = os.path.join(model_path, "UDR-3_0_60_40")
        if not os.path.exists(logs_dir):
            continue
        
        log_files = os.listdir(logs_dir)
        
        # Filter by sample traces if available
        if use_sample_filter:
            log_files = [f for f in log_files if f in sample_traces]
        
        # Compute rewards for all valid logs
        rewards = []
        valid_traces = []
        pensieve_rewards_for_model = []
        
        for log_file in log_files:
            log_path = os.path.join(logs_dir, log_file)
            if is_valid_log(log_path):
                reward = compute_mean_reward(log_path)
                if reward is not None:
                    rewards.append(reward)
                    valid_traces.append(log_file)
                    # Get corresponding Pensieve reward if available
                    if log_file in pensieve_baseline:
                        pensieve_rewards_for_model.append(pensieve_baseline[log_file])
        
        if rewards:
            config['mean_reward'] = mean(rewards)
            config['std_reward'] = std(rewards)
            config['num_traces'] = len(rewards)
            config['valid_traces'] = valid_traces
            config['model_dir'] = model_dir
            
            # Calculate improvement over Pensieve baseline
            if pensieve_rewards_for_model:
                config['pensieve_baseline_mean'] = mean(pensieve_rewards_for_model)
                # Only calculate improvement for traces that have both UNUM and Pensieve results
                improvements = []
                for i, trace in enumerate(valid_traces):
                    if trace in pensieve_baseline:
                        pensieve_reward = pensieve_baseline[trace]
                        unum_reward = rewards[i]
                        if pensieve_reward != 0:
                            improvement = (unum_reward - pensieve_reward) / abs(pensieve_reward)
                            improvements.append(improvement)
                
                if improvements:
                    config['mean_improvement'] = mean(improvements)
                    config['improvement_traces'] = len(improvements)
                else:
                    config['mean_improvement'] = None
                    config['improvement_traces'] = 0
            else:
                config['pensieve_baseline_mean'] = None
                config['mean_improvement'] = None
                config['improvement_traces'] = 0
            
            models_data[model_dir] = config
            
            improvement_str = f", improvement: {config['mean_improvement']:.3f}" if config['mean_improvement'] is not None else ", no baseline"
            print(f"Processed {model_dir}: {len(rewards)} traces, mean reward: {mean(rewards):.3f}{improvement_str}")
    
    return models_data, pensieve_baseline

def create_results_table(models_data, pensieve_baseline):
    """Create results table organized by context window and hidden size."""
    
    # Group results by (context_window, hidden_size)
    grouped_results = defaultdict(list)
    
    for model_name, config in models_data.items():
        key = (config['context_window'], config['hidden_size'])
        grouped_results[key].append(config)
    
    # Find best model for each configuration
    best_results = {}
    for key, configs in grouped_results.items():
        # Find the configuration with highest mean reward
        best_config = max(configs, key=lambda x: x['mean_reward'])
        best_results[key] = best_config
    
    # Get unique context windows and hidden sizes for table structure
    context_windows = sorted(set(key[0] for key in best_results.keys()))
    hidden_sizes = sorted(set(key[1] for key in best_results.keys()))
    
    print(f"Context windows found: {context_windows}")
    print(f"Hidden sizes found: {hidden_sizes}")
    
    # Calculate Pensieve baseline average for each configuration
    pensieve_by_config = {}
    if pensieve_baseline:
        # For now, use overall Pensieve average since we don't have config-specific baselines
        overall_pensieve_mean = mean(list(pensieve_baseline.values()))
        for cw in context_windows:
            for hs in hidden_sizes:
                pensieve_by_config[(cw, hs)] = overall_pensieve_mean
    
    # Create table data
    table_data = []
    headers = ['Context Window \\ Hidden Size'] + [f'H{hs}' for hs in hidden_sizes]
    
    # UNUM Results Row
    for cw in context_windows:
        row = [f'CW{cw} (UNUM)']
        for hs in hidden_sizes:
            key = (cw, hs)
            if key in best_results:
                config = best_results[key]
                # Format: mean_reward ± std (num_traces) [model_info]
                cell_value = f"{config['mean_reward']:.3f} ± {config['std_reward']:.3f}"
                cell_value += f" ({config['num_traces']} traces)"
                cell_value += f"\n[S{config['seed']}, E{config['epoch']}]"
                row.append(cell_value)
            else:
                row.append('N/A')
        table_data.append(row)
    
    # Pensieve Baseline Row (if available)
    if pensieve_baseline:
        for cw in context_windows:
            row = [f'CW{cw} (Pensieve)']
            for hs in hidden_sizes:
                key = (cw, hs)
                if key in pensieve_by_config:
                    baseline_reward = pensieve_by_config[key]
                    cell_value = f"{baseline_reward:.3f}"
                    cell_value += f"\n[Baseline]"
                    row.append(cell_value)
                else:
                    row.append('N/A')
            table_data.append(row)
    
    # Improvement Row (if available)
    if pensieve_baseline:
        for cw in context_windows:
            row = [f'CW{cw} (Improvement %)']
            for hs in hidden_sizes:
                key = (cw, hs)
                if key in best_results and best_results[key]['mean_improvement'] is not None:
                    improvement = best_results[key]['mean_improvement'] * 100
                    cell_value = f"{improvement:+.2f}%"
                    cell_value += f"\n({best_results[key]['improvement_traces']} traces)"
                    row.append(cell_value)
                else:
                    row.append('N/A')
            table_data.append(row)
    
    return table_data, headers, best_results, pensieve_by_config

def print_detailed_results(best_results, pensieve_by_config):
    """Print detailed results for each configuration."""
    print("\n" + "="*80)
    print("DETAILED RESULTS BY CONFIGURATION")
    print("="*80)
    
    for key, config in sorted(best_results.items()):
        cw, hs = key
        print(f"\nContext Window {cw}, Hidden Size {hs}:")
        print(f"  Best Model: {config['model_dir']}")
        print(f"  Mean Reward: {config['mean_reward']:.4f} ± {config['std_reward']:.4f}")
        print(f"  Seed: {config['seed']}, Epoch: {config['epoch']}")
        print(f"  Input Type: {config['input_type']}")
        print(f"  Number of Traces: {config['num_traces']}")
        print(f"  Adaptor Hidden: {config.get('adaptor_hidden', 'N/A')}")
        print(f"  Adaptor CW: {config.get('adaptor_cw', 'N/A')}")
        
        if config.get('pensieve_baseline_mean') is not None:
            print(f"  Pensieve Baseline: {config['pensieve_baseline_mean']:.4f}")
        if config.get('mean_improvement') is not None:
            print(f"  Improvement over Pensieve: {config['mean_improvement']*100:+.2f}% ({config['improvement_traces']} traces)")
        else:
            print(f"  Improvement over Pensieve: N/A")

def get_04_20_model_results():
    """Get results from the 04_20_model_set used in boxplot for comparison."""
    old_results_dir = "/home/jane/Genet/scripts/04_20_model_set/04_20_model_set/04_20_model_set"
    target_model_name = "action_256_20"  # From plot_reward_boxplot.py
    
    if not os.path.exists(old_results_dir):
        print(f"Warning: 04_20_model_set directory not found: {old_results_dir}")
        return None
    
    # Get sample traces for consistent comparison
    sample_traces = get_sample_traces()
    pensieve_baseline = get_pensieve_baseline_rewards(sample_traces)
    
    if not sample_traces or not pensieve_baseline:
        print("Warning: No sample traces or pensieve baseline for 04_20_model_set comparison")
        return None
    
    # Look for the target model directory
    model_dirs = [d for d in os.listdir(old_results_dir) if d.startswith("server_")]
    
    for model_dir in model_dirs:
        # Use the same mapping logic from plot_reward_boxplot.py
        server_id = int(model_dir.split('_')[1])
        adaptor_configs = []
        for s in [10, 20, 30, 40, 50]:
            for inp in ["original_selection", "hidden_state"]:
                for hl in [128, 256]:
                    adaptor_configs.append((inp, hl, s))
        
        if server_id <= len(adaptor_configs):
            index = (server_id - 1) % len(adaptor_configs)
            adaptor_input, hidden_layer, seed = adaptor_configs[index]
            model_name = f"{'action' if adaptor_input == 'original_selection' else 'hidden'}_{hidden_layer}_{seed}"
            
            if model_name == target_model_name:
                # Found the target model, get its results
                model_path = os.path.join(old_results_dir, model_dir, "UDR-3_0_60_40")
                if os.path.exists(model_path):
                    rewards = []
                    improvements = []
                    
                    for trace_file in sample_traces:
                        trace_path = os.path.join(model_path, trace_file)
                        if os.path.exists(trace_path) and is_valid_log(trace_path):
                            reward = compute_mean_reward(trace_path)
                            if reward is not None and trace_file in pensieve_baseline:
                                rewards.append(reward)
                                pensieve_reward = pensieve_baseline[trace_file]
                                if pensieve_reward != 0:
                                    improvement = (reward - pensieve_reward) / abs(pensieve_reward)
                                    improvements.append(improvement)
                    
                    if improvements:
                        return {
                            'mean_reward': mean(rewards),
                            'mean_improvement': mean(improvements),
                            'num_traces': len(improvements),
                            'model_name': model_name
                        }
    
    return None

def create_improvement_csv(best_results, context_windows, hidden_sizes, output_file):
    """Create a simplified CSV showing only improvement percentages."""
    
    # Get 04_20_model_set results for comparison
    old_model_results = get_04_20_model_results()
    
    with open(output_file, 'w') as f:
        # Write header
        header = ['Context_Window\\Hidden_Size'] + [f'H{hs}' for hs in hidden_sizes]
        f.write(','.join(header) + '\n')
        
        # Write improvement rows for each context window
        for cw in context_windows:
            row = [f'CW{cw}']
            for hs in hidden_sizes:
                key = (cw, hs)
                if key in best_results and best_results[key]['mean_improvement'] is not None:
                    improvement_pct = best_results[key]['mean_improvement'] * 100
                    row.append(f'{improvement_pct:+.2f}%')
                else:
                    row.append('N/A')
            f.write(','.join(row) + '\n')
        
        # Add 04_20_model_set results row if available
        if old_model_results:
            row = ['04_20_model_action_256_20']
            improvement_pct = old_model_results['mean_improvement'] * 100
            # Add this value to all columns since it's a single model result
            for hs in hidden_sizes:
                row.append(f'{improvement_pct:+.2f}%')
            f.write(','.join(row) + '\n')
        
        # Add pensieve baseline row for reference
        row = ['Pensieve_Baseline']
        for hs in hidden_sizes:
            row.append('0.00%')  # Baseline is 0% improvement by definition
        f.write(','.join(row) + '\n')

def main():
    print("Analyzing test results by context window and hidden size...")
    print(f"Reading from: {TEST_RESULTS_DIR}")
    
    # Collect all model results
    models_data, pensieve_baseline = collect_model_results()
    
    if not models_data:
        print("No valid model data found!")
        return
    
    print(f"\nProcessed {len(models_data)} models total")
    
    # Create results table
    table_data, headers, best_results, pensieve_by_config = create_results_table(models_data, pensieve_baseline)
    
    # Get dimensions for CSV
    context_windows = sorted(set(key[0] for key in best_results.keys()))
    hidden_sizes = sorted(set(key[1] for key in best_results.keys()))
    
    # Print the main results table
    print("\n" + "="*80)
    print("BEST PERFORMING MODELS BY CONFIGURATION")
    print("="*80)
    
    # Simple table formatting since tabulate might not be available
    print(f"{'':25}", end="")
    for header in headers[1:]:
        print(f"{header:20}", end="")
    print()
    print("-" * (25 + 20 * len(headers[1:])))
    
    for row in table_data:
        print(f"{row[0]:25}", end="")
        for cell in row[1:]:
            # Format multi-line cells for simple display
            cell_lines = str(cell).split('\n')
            print(f"{cell_lines[0]:20}", end="")
        print()
        # Print second line if exists
        has_second_line = any('\n' in str(cell) for cell in row[1:])
        if has_second_line:
            print(f"{'':25}", end="")
            for cell in row[1:]:
                cell_lines = str(cell).split('\n')
                second_line = cell_lines[1] if len(cell_lines) > 1 else ""
                print(f"{second_line:20}", end="")
            print()
    
    # Print detailed results
    print_detailed_results(best_results, pensieve_by_config)
    
    # Save simplified results to CSV
    output_file = os.path.join(RESULTS_BASE_DIR, "results_improvement_table.csv")
    
    # Create simplified improvement table
    create_improvement_csv(best_results, context_windows, hidden_sizes, output_file)
    
    print(f"\nSimplified improvement table saved to: {output_file}")

if __name__ == "__main__":
    main()
