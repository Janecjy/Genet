import os
import pandas as pd
import matplotlib.pyplot as plt

# Define result directories
RESULTS_DIR = "/home/jane/Genet/scripts/03_22_model_set"
TRACE_COMPARISON_DIR = os.path.join(RESULTS_DIR, "trace_comparisons")
os.makedirs(TRACE_COMPARISON_DIR, exist_ok=True)  # Ensure directory exists

PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/synthetic_test_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "03_22_model_set/03_22_model_set")

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

# Function to read a log file and compute mean reward (excluding the first row)
def compute_mean_reward(log_path):
    try:
        df = pd.read_csv(log_path, delim_whitespace=True)
        if 'reward' in df.columns and len(df) > 1:
            return df['reward'][1:].mean()  # Exclude the first row
        else:
            return None
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None

# Step 1: Collect all available log files for Pensieve-original
pensieve_logs = {file: os.path.join(PENSIEVE_DIR, file) for file in os.listdir(PENSIEVE_DIR) if file.startswith("log_RL_trace_")}
unum_models = {model: os.path.join(UNUM_DIR, model) for model in os.listdir(UNUM_DIR) if model.startswith("server_")}

# Step 2: Process each trace separately
for trace_file, pensieve_log_path in pensieve_logs.items():
    trace_name = trace_file.replace("log_", "").replace(".txt", "")

    model_rewards = {"pensieve-original": compute_mean_reward(pensieve_log_path)}
    action_models = []
    hidden_models = []

    # Step 3: Find all models that have logs for this trace
    for model, model_path in unum_models.items():
        server_id = int(model.split('_')[1])  # Extract server number
        model_name = get_model_config(server_id)  # Get renamed model config
        trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)

        if os.path.exists(trace_path):
            reward = compute_mean_reward(trace_path)
            if reward is not None:
                model_rewards[model_name] = reward
                if model_name.startswith("action"):
                    action_models.append((model_name, reward))
                else:
                    hidden_models.append((model_name, reward))

    # Step 4: Sort action & hidden models
    action_models.sort()
    hidden_models.sort()

    # Step 5: Generate x-axis labels and mean reward values
    x_labels = ["pensieve-original"] + [m[0] for m in action_models] + [m[0] for m in hidden_models]
    mean_rewards = [model_rewards[label] for label in x_labels]

    if len(x_labels) > 1:

        # Step 6: Plot the results
        plt.figure(figsize=(12, 6))
        plt.bar(x_labels, mean_rewards, color=['blue'] + ['orange'] * len(action_models) + ['green'] * len(hidden_models))
        plt.xlabel("Model (Adaptor Input_Hidden_Size_Seed)")
        plt.ylabel("Mean Reward")
        plt.title(f"Comparison of Mean Rewards for {trace_name}")
        plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
        plt.tight_layout()
        plt.ylim(-5, 3)  # Set y-axis limits

        # Step 7: Save each trace's plot separately
        plot_path = os.path.join(TRACE_COMPARISON_DIR, f"{trace_name}_comparison.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")

print("All per-trace comparison plots generated successfully.")
