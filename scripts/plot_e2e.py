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

# Step 2: Find common traces across all models
common_traces = set(pensieve_logs.keys())

for model, model_path in unum_models.items():
    server_id = int(model.split('_')[1])  # Extract server number
    model_name = get_model_config(server_id)  # Get renamed model config
    model_traces = set()

    for trace_file in pensieve_logs.keys():
        trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
        if os.path.exists(trace_path):
            model_traces.add(trace_file)

    common_traces &= model_traces  # Keep only traces that exist in all models

# Step 3: Compute mean metrics per model across common traces
model_metrics = {"pensieve-original": []}
action_models = {}
hidden_models = {}

for trace_file in common_traces:
    pensieve_rebuffer, pensieve_bitrate = compute_metrics(pensieve_logs[trace_file])
    if pensieve_rebuffer is not None and pensieve_bitrate is not None:
        model_metrics["pensieve-original"].append((pensieve_rebuffer, pensieve_bitrate))

    for model, model_path in unum_models.items():
        server_id = int(model.split('_')[1])  # Extract server number
        model_name = get_model_config(server_id)
        trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)

        if os.path.exists(trace_path):
            rebuffer_ratio, avg_bitrate = compute_metrics(trace_path)
            if rebuffer_ratio is not None and avg_bitrate is not None:
                model_metrics.setdefault(model_name, []).append((rebuffer_ratio, avg_bitrate))

# Step 4: Compute mean values for each model
mean_metrics = {}
for model_name, values in model_metrics.items():
    if values:
        mean_rebuffer = np.mean([v[0] for v in values])
        mean_bitrate = np.mean([v[1] for v in values])
        mean_metrics[model_name] = (mean_rebuffer, mean_bitrate)
        if model_name.startswith("action"):
            action_models[model_name] = (mean_rebuffer, mean_bitrate)
        elif model_name.startswith("hidden"):
            hidden_models[model_name] = (mean_rebuffer, mean_bitrate)

# Step 5: Plot Mean Scatter Plot
plt.figure(figsize=(8, 6))

# Plot Pensieve-original
if "pensieve-original" in mean_metrics:
    x, y = mean_metrics["pensieve-original"]
    plt.scatter(x, y, color='blue', label="Pensieve-original", s=100, edgecolors='black', marker="o")

# Plot action models
for model_name, (x, y) in sorted(action_models.items()):
    plt.scatter(x, y, color='orange', label="Action" if "Action" not in plt.gca().get_legend_handles_labels()[1] else "", s=80, alpha=0.8)

# Plot hidden models
for model_name, (x, y) in sorted(hidden_models.items()):
    plt.scatter(x, y, color='green', label="Hidden" if "Hidden" not in plt.gca().get_legend_handles_labels()[1] else "", s=80, alpha=0.8)

# Labels and Title
plt.xlabel("Mean 90th Percentile Rebuffering Time (ms)")
plt.ylabel("Mean Average Bitrate (Kbps)")
# plt.title("Mean Scatter Plot Across All Traces")
plt.legend()
plt.grid(True)

# Save the scatter plot
plot_path = os.path.join(SCATTER_PLOT_DIR, "mean_scatter.png")
plt.savefig(plot_path)
plt.close()

print(f"Saved mean scatter plot: {plot_path}")
print("Mean scatter plot generated successfully.")
