import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Directories ---------------------
RESULTS_DIR = "/home/jane/Genet/scripts/04_20_model_set"
PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/testing_trace_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "04_20_model_set/04_20_model_set")

# ---------------- Configurations ---------------------
adaptor_inputs = ["original_selection", "hidden_state"]
adaptor_hidden_layers = [128, 256]
seeds = [10, 20, 30, 40, 50]
target_model_name = "action_256_20"

adaptor_configs = [(inp, hl, s) for s in seeds for inp in adaptor_inputs for hl in adaptor_hidden_layers]

def get_model_config(server_id):
    index = (server_id - 1) % len(adaptor_configs)
    adaptor_input, hidden_layer, seed = adaptor_configs[index]
    return f"{'action' if adaptor_input == 'original_selection' else 'hidden'}_{hidden_layer}_{seed}"

def is_valid_log(log_path):
    try:
        df = pd.read_csv(log_path, delim_whitespace=True)
        return 'reward' in df.columns and len(df) > 2
    except:
        return False

def compute_mean_reward(log_path):
    try:
        df = pd.read_csv(log_path, delim_whitespace=True)
        if 'reward' in df.columns and len(df) > 1:
            return df['reward'][1:].mean()
    except:
        return None
    return None

def get_trace_filter_prefix(trace_type):
    return f"log_RL_{trace_type}-test_"

# ----------------- Compute ratio for each trace type ----------------------
ratios = {}
for trace_type in ['fcc', 'norway']:
    prefix = get_trace_filter_prefix(trace_type)

    # Collect Pensieve traces
    pensieve_logs = {
        file: os.path.join(PENSIEVE_DIR, file)
        for file in os.listdir(PENSIEVE_DIR)
        if file.startswith(prefix)
    }

    # Collect Unum models
    unum_models = {
        model: os.path.join(UNUM_DIR, model)
        for model in os.listdir(UNUM_DIR) if model.startswith("server_")
    }

    # Get model config map
    model_config_map = {}
    for model, model_path in unum_models.items():
        server_id = int(model.split('_')[1])
        model_name = get_model_config(server_id)
        model_config_map[model] = model_name

    # Identify common traces available across all servers
    common_traces = set(pensieve_logs.keys())
    for model, model_path in unum_models.items():
        for trace_file in list(common_traces):
            trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
            if not os.path.exists(trace_path) or not is_valid_log(trace_path):
                common_traces.discard(trace_file)

    if not common_traces:
        print(f"No common traces for {trace_type}")
        continue

    # Compute Pensieve rewards
    pensieve_rewards = []
    for trace_file in common_traces:
        reward = compute_mean_reward(pensieve_logs[trace_file])
        if reward is not None:
            pensieve_rewards.append(reward)

    pensieve_mean = sum(pensieve_rewards) / len(pensieve_rewards) if pensieve_rewards else None

    # Compute Unum-Adptor rewards (only action_256_20)
    unum_rewards = []
    for model, model_path in unum_models.items():
        if model_config_map[model] != target_model_name:
            continue
        for trace_file in common_traces:
            trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
            reward = compute_mean_reward(trace_path)
            if reward is not None:
                unum_rewards.append(reward)

    unum_mean = sum(unum_rewards) / len(unum_rewards) if unum_rewards else None
    print(unum_mean, pensieve_mean)

    # Store ratio
    if pensieve_mean and unum_mean:
        ratios[trace_type.upper()] = (unum_mean-pensieve_mean)/abs(pensieve_mean)*100

# ----------------- Plot ----------------------
plt.figure(figsize=(4, 5))
x = np.arange(len(ratios.keys())) * 0.4 # <--- Multiply by a spacing factor < 1 to reduce the gap
bar_width = 0.3
bars = plt.bar(x, ratios.values(), color=['#1f77b4', '#ff7f0e'], width=bar_width)

plt.ylabel("Improvement (%)", fontsize=20)
plt.xticks(x, ratios.keys(), fontsize=20)
plt.yticks(fontsize=20)

for bar, val in zip(bars, ratios.values()):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{val:.2f}%", 
                ha='center', va='bottom', fontsize=20)


# plt.title("Unum-Adptor vs Pensieve Mean Reward Ratio", fontsize=14)
# plt.grid(axis='y')
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "reward_improvement_fcc_norway.png")
plt.savefig(plot_path)
# plt.show()

print(f"Saved ratio plot at: {plot_path}")

# height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{val:.2%}", 
#                  ha='center', va='bottom', fontsize=20)
