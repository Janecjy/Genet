import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Directories ---------------------
RESULTS_DIR = "/home/jane/Genet/scripts/04_20_model_set"
PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/testing_trace_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "04_20_model_set/04_20_model_set")
target_model_name = "action_256_20"

# ------------------ Model Config Mapping ---------------------
adaptor_inputs = ["original_selection", "hidden_state"]
adaptor_hidden_layers = [128, 256]
seeds = [10, 20, 30, 40, 50]
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

# ----------------- Compute trace-by-trace improvement ----------------------
for trace_type in ['fcc', 'norway']:
    prefix = get_trace_filter_prefix(trace_type)

    pensieve_logs = {
        file: os.path.join(PENSIEVE_DIR, file)
        for file in os.listdir(PENSIEVE_DIR)
        if file.startswith(prefix)
    }

    unum_models = {
        model: os.path.join(UNUM_DIR, model)
        for model in os.listdir(UNUM_DIR) if model.startswith("server_")
    }

    model_config_map = {
        model: get_model_config(int(model.split('_')[1]))
        for model in unum_models
    }

    common_traces = set(pensieve_logs.keys())
    for model, model_path in unum_models.items():
        for trace_file in list(common_traces):
            trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
            if not os.path.exists(trace_path) or not is_valid_log(trace_path):
                common_traces.discard(trace_file)

    if not common_traces:
        print(f"No common traces for {trace_type}")
        continue

    # ------------------ Timeline Data ---------------------
    improvement_list = []
    trace_names = []

    for trace_file in sorted(common_traces):
        pensieve_reward = compute_mean_reward(pensieve_logs[trace_file])
        unum_reward = None

        for model, model_path in unum_models.items():
            if model_config_map[model] == target_model_name:
                trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
                unum_reward = compute_mean_reward(trace_path)
                if unum_reward is not None:
                    break

        if pensieve_reward is not None and unum_reward is not None:
            improvement = (unum_reward - pensieve_reward) / abs(pensieve_reward) * 100
            improvement_list.append(improvement)
            trace_names.append(trace_file.replace(".txt", ""))

    # ------------------ Plot Timeline ---------------------
    plt.figure(figsize=(10, 4))
    x = np.arange(len(improvement_list))
    plt.plot(x, improvement_list, 'o-', linewidth=2, label=f"{trace_type.upper()}")

    plt.xticks(x, trace_names, rotation=90, fontsize=8)
    plt.ylabel("Reward Improvement (%)", fontsize=14)
    plt.xlabel("Trace", fontsize=14)
    plt.title(f"Reward Improvement over Time - {trace_type.upper()}", fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    timeline_plot_path = os.path.join(RESULTS_DIR, f"reward_improvement_timeline_{trace_type}.png")
    plt.savefig(timeline_plot_path)
    print(f"Saved timeline plot at: {timeline_plot_path}")
