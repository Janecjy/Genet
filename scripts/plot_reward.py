import os
import pandas as pd
import matplotlib.pyplot as plt

# Define result directories
RESULTS_DIR = "/home/jane/Genet/scripts/03_22_model_set"
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

# Function to check if log file has more than 2 rows
def is_valid_log(log_path):
    try:
        df = pd.read_csv(log_path, delim_whitespace=True)
        return 'reward' in df.columns and len(df) > 2
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return False

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

# Step 1: Collect all available log files
pensieve_logs = {file: os.path.join(PENSIEVE_DIR, file) for file in os.listdir(PENSIEVE_DIR) if file.startswith("log_RL_trace_")}
unum_models = {model: os.path.join(UNUM_DIR, model) for model in os.listdir(UNUM_DIR) if model.startswith("server_")}

# Step 2: Find common traces across all models
common_traces = set(pensieve_logs.keys())
model_rewards = {}

for model, model_path in unum_models.items():
    server_id = int(model.split('_')[1])
    model_name = get_model_config(server_id)
    model_rewards[model] = {"name": model_name, "path": model_path, "rewards": []}
    model_traces = {trace for trace in pensieve_logs.keys()
                     if os.path.exists(os.path.join(model_path, "UDR-3_0_60_40", trace)) and is_valid_log(os.path.join(model_path, "UDR-3_0_60_40", trace))}
    common_traces &= model_traces

print("Common traces:", common_traces)

# Step 3: Compute mean rewards per model
pensieve_mean_rewards = []
unum_mean_rewards = {}

for trace_file in common_traces:
    # Compute Pensieve-original mean reward
    pensieve_reward = compute_mean_reward(pensieve_logs[trace_file])
    if pensieve_reward is not None:
        pensieve_mean_rewards.append(pensieve_reward)

    # Compute Unum-adaptor model rewards
    for model, model_data in model_rewards.items():
        trace_path = os.path.join(model_data["path"], "UDR-3_0_60_40", trace_file)
        if os.path.exists(trace_path):
            reward = compute_mean_reward(trace_path)
            if reward is not None:
                unum_mean_rewards.setdefault(model, []).append(reward)

# Step 4: Compute final mean rewards
pensieve_final_mean = sum(pensieve_mean_rewards) / len(pensieve_mean_rewards) if pensieve_mean_rewards else None
unum_final_means = {model: sum(rewards) / len(rewards) for model, rewards in unum_mean_rewards.items() if rewards}

# Step 5: Map server_X to readable names and sort them
x_labels = ["pensieve-original"]
mean_rewards = [pensieve_final_mean]

# Sort the action and hidden models separately
action_models = []
hidden_models = []

for model in unum_final_means.keys():
    model_name = model_rewards[model]["name"]
    if model_name.startswith("action"):
        action_models.append((model_name, unum_final_means[model]))
    else:
        hidden_models.append((model_name, unum_final_means[model]))

# Sort both groups alphabetically
action_models.sort()
hidden_models.sort()

# Append sorted models
for model_name, mean_reward in action_models:
    x_labels.append(model_name)
    mean_rewards.append(mean_reward)

for model_name, mean_reward in hidden_models:
    x_labels.append(model_name)
    mean_rewards.append(mean_reward)

# Step 6: Plot the results
print("Mean rewards:", mean_rewards)
print("X labels:", x_labels)
plt.figure(figsize=(12, 6))
plt.bar(x_labels, mean_rewards, color=['blue'] + ['orange'] * len(action_models) + ['green'] * len(hidden_models))
plt.xlabel("Model (Adaptor Input_Hidden_Size_Seed)")
plt.ylabel("Mean Reward")
plt.title("Comparison of Mean Rewards Across Models")
plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
plt.ylim(-5, 3)
plt.tight_layout()

# Save and display the plot
plt.savefig(os.path.join(RESULTS_DIR, "reward_comparison.png"))
plt.show()
