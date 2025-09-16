import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Directories ---------------------
RESULTS_DIR = "/home/jane/Genet/scripts/04_20_model_set"
PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/testing_trace_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "04_20_model_set/04_20_model_set")
SAMPLE_DIR = "/home/jane/Genet/abr_trace/testing_trace_mahimahi_sample"

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

# ----------------- Collect data ----------------------
all_improvements = []
labels = []

for trace_type in ['fcc', 'norway']:
    prefix = get_trace_filter_prefix(trace_type)

    # Get traces from sample directory
    sample_traces = [
        f for f in os.listdir(SAMPLE_DIR) 
        if f.startswith(f"{trace_type}-test")
    ]
    
    if not sample_traces:
        print(f"No sample traces for {trace_type}")
        continue
    
    print(f"[{trace_type.upper()}] Found {len(sample_traces)} sample traces")

    # Collect Pensieve traces (only for sample traces)
    pensieve_logs = {}
    for trace_file in sample_traces:
        pensieve_trace_name = f"log_RL_{trace_file}"
        pensieve_path = os.path.join(PENSIEVE_DIR, pensieve_trace_name)
        if os.path.exists(pensieve_path):
            pensieve_logs[pensieve_trace_name] = pensieve_path

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

    # Identify common traces available across all servers (only from sample traces)
    common_traces = set(pensieve_logs.keys())
    for model, model_path in unum_models.items():
        for trace_file in list(common_traces):
            trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
            if not os.path.exists(trace_path) or not is_valid_log(trace_path):
                common_traces.discard(trace_file)

    if not common_traces:
        print(f"No common traces for {trace_type}")
        continue

    # Collect per-trace rewards and compute improvements
    improvements = []
    pensieve_filtered_rewards = []
    unum_filtered_rewards = []
    filtered_traces = []

    for trace_file in common_traces:
        pensieve_reward = compute_mean_reward(pensieve_logs[trace_file])
        unum_reward = None
        for model, model_path in unum_models.items():
            if model_config_map[model] != target_model_name:
                continue
            trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
            if os.path.exists(trace_path):
                unum_reward = compute_mean_reward(trace_path)
                break
        if pensieve_reward is not None and unum_reward is not None and pensieve_reward != 0:
            improvement = (unum_reward - pensieve_reward) / abs(pensieve_reward)
            improvements.append(improvement)
            pensieve_filtered_rewards.append(pensieve_reward)
            unum_filtered_rewards.append(unum_reward)
            filtered_traces.append(trace_file)

    if improvements:
        print(f"[{trace_type.upper()}] Found {len(improvements)} valid sample traces")
        print(f"[{trace_type.upper()}] Sample traces processed:")
        for i, trace in enumerate(filtered_traces):
            print(f"  {i+1}. {trace}")
        all_improvements.append(np.array(improvements))
        print(f"[{trace_type.upper()}] Mean Pensieve Reward improvement: {np.mean(improvements):.4f}")
        labels.append(trace_type.upper())

    # Print mean rewards for sample traces
    if pensieve_filtered_rewards and unum_filtered_rewards:
        mean_pensieve = np.mean(pensieve_filtered_rewards)
        mean_unum = np.mean(unum_filtered_rewards)
        print(f"[{trace_type.upper()}] Mean Pensieve Reward (sample traces): {mean_pensieve:.4f}")
        print(f"[{trace_type.upper()}] Mean UNUM-Adaptor Reward (sample traces): {mean_unum:.4f}")

print(all_improvements)

# ----------------- Plot boxplot ----------------------
plt.figure(figsize=(3, 4))
box = plt.boxplot(
    [imp * 100 for imp in all_improvements],  # convert to %
    labels=labels,
    patch_artist=True,
    widths=0.6,  # Make boxes wider (default is 0.5)
    boxprops=dict(linewidth=1.5),
    medianprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker='o', markersize=4, linestyle='none')
)

# Print the mean value for each box
for i, improvements in enumerate(all_improvements):
    mean = np.mean(improvements) * 100
    print(f"Mean improvement for {labels[i]}: {mean:.2f}%")

plt.ylabel("Reward Improvement (%)", fontsize=22, y=0.4)  # Increase labelpad to move label away from axis
plt.xticks(fontsize=22, rotation=10)
plt.yticks(fontsize=22)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plot_path = os.path.join(RESULTS_DIR, "reward_improvement_fcc_norway_sample_boxplot.png")
plot_pdf_path = os.path.join(RESULTS_DIR, "reward_improvement_fcc_norway_sample_boxplot.pdf")
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
plt.savefig(plot_pdf_path, bbox_inches='tight', pad_inches=0.1)
# plt.show()

print(f"Saved boxplot at: {plot_path}")
