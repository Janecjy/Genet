import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define directories
RESULTS_DIR = "/home/jane/Genet/scripts/03_19_model_set"
SCATTER_PLOT_DIR = os.path.join(RESULTS_DIR, "scatter_plots")
os.makedirs(SCATTER_PLOT_DIR, exist_ok=True)  # Ensure directory exists

PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/synthetic_test_plus_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "03_19_model_set/03_19_model_set")

# Define model configurations (same as training script)
adaptor_inputs = ["original_selection", "hidden_state"]
adaptor_hidden_layers = [128, 256]
seeds = [10, 20, 30, 40, 50]

# Generate (input, hidden, seed) mappings for each server
adaptor_configs = []
for seed in seeds:
    for input_type in adaptor_inputs:
        for hidden_layer in adaptor_hidden_layers:
            adaptor_configs.append((input_type, hidden_layer, seed))

def get_model_config(server_id):
    """Maps server ID to corresponding (input, hidden, seed) configuration."""
    index = (server_id - 1) % len(adaptor_configs)
    adaptor_input, hidden_layer, seed = adaptor_configs[index]
    model_name = f"{'action' if adaptor_input == 'original_selection' else 'hidden'}_{hidden_layer}_{seed}"
    return model_name

# Function to read a log file and compute 90th percentile rebuffering ratio and average bitrate
def compute_metrics(log_path):
    try:
        df = pd.read_csv(log_path, delim_whitespace=True)
        if 'rebuffer_time' in df.columns and 'bit_rate' in df.columns and len(df) > 1:
            # Compute 90th percentile rebuffering ratio
            rebuffer_ratio_90th = np.percentile(df['rebuffer_time'][1:], 90)
            # Compute average bitrate
            avg_bitrate = df['bit_rate'][1:].mean()
            return rebuffer_ratio_90th, avg_bitrate
        else:
            return None, None
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None, None

# Step 1: Collect all available log files for Pensieve-original
pensieve_logs = {file: os.path.join(PENSIEVE_DIR, file) for file in os.listdir(PENSIEVE_DIR) if file.startswith("log_RL_trace_")}
unum_models = {model: os.path.join(UNUM_DIR, model) for model in os.listdir(UNUM_DIR) if model.startswith("server_")}

# Step 2: Process each trace separately
for trace_file, pensieve_log_path in pensieve_logs.items():
    trace_name = trace_file.replace("log_", "").replace(".txt", "")

    model_metrics = {"pensieve-original": compute_metrics(pensieve_log_path)}
    action_models = []
    hidden_models = []

    # Step 3: Find all models that have logs for this trace
    for model, model_path in unum_models.items():
        server_id = int(model.split('_')[1])  # Extract server number
        model_name = get_model_config(server_id)  # Get renamed model config
        trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)

        if os.path.exists(trace_path):
            rebuffer_ratio, avg_bitrate = compute_metrics(trace_path)
            if rebuffer_ratio is not None and avg_bitrate is not None:
                model_metrics[model_name] = (rebuffer_ratio, avg_bitrate)
                if model_name.startswith("action"):
                    action_models.append((model_name, rebuffer_ratio, avg_bitrate))
                else:
                    hidden_models.append((model_name, rebuffer_ratio, avg_bitrate))

    # Step 4: Plot Scatter Plot
    plt.figure(figsize=(8, 6))
    
    # Plot Pensieve-original
    if "pensieve-original" in model_metrics:
        x, y = model_metrics["pensieve-original"]
        plt.scatter(x, y, color='blue', label="Pensieve-original", s=100, edgecolors='black', marker="o")

    # Plot action models
    for model_name, x, y in action_models:
        plt.scatter(x, y, color='orange', label="Action" if "Action" not in plt.gca().get_legend_handles_labels()[1] else "", s=80, alpha=0.8)

    # Plot hidden models
    for model_name, x, y in hidden_models:
        plt.scatter(x, y, color='green', label="Hidden" if "Hidden" not in plt.gca().get_legend_handles_labels()[1] else "", s=80, alpha=0.8)

    # Labels and Title
    plt.xlabel("90th Percentile Rebuffering Ratio")
    plt.ylabel("Average Bitrate")
    plt.title(f"Scatter Plot for {trace_name}")
    plt.legend()
    plt.grid(True)
    
    # Save the scatter plot
    plot_path = os.path.join(SCATTER_PLOT_DIR, f"{trace_name}_scatter.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved scatter plot: {plot_path}")

print("All per-trace scatter plots generated successfully.")
