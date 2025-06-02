import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Directories ---------------------
RESULTS_DIR = "/home/jane/Genet/scripts/04_20_model_set"
SCATTER_PLOT_DIR = os.path.join(RESULTS_DIR, "scatter_plots")
os.makedirs(SCATTER_PLOT_DIR, exist_ok=True)

PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/testing_trace_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "04_20_model_set/04_20_model_set")

adaptor_inputs = ["original_selection", "hidden_state"]
adaptor_hidden_layers = [128, 256]
seeds = [10, 20, 30, 40, 50]
adaptor_configs = [(inp, hl, s) for s in seeds for inp in adaptor_inputs for hl in adaptor_hidden_layers]

def get_model_config(server_id):
    index = (server_id - 1) % len(adaptor_configs)
    adaptor_input, hidden_layer, seed = adaptor_configs[index]
    return f"{'action' if adaptor_input == 'original_selection' else 'hidden'}_{hidden_layer}_{seed}"

def compute_metrics(log_path):
    try:
        df = pd.read_csv(log_path, delim_whitespace=True)
        if 'rebuffer_time' in df.columns and 'bit_rate' in df.columns and len(df) > 1:
            rebuffer_ratio_90th = np.percentile(df['rebuffer_time'][1:], 90)
            avg_bitrate = df['bit_rate'][1:].mean()
            avg_rebuffer = df['rebuffer_time'][1:].mean()
            return (avg_rebuffer, rebuffer_ratio_90th, avg_bitrate)
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return None, None, None

# ---------------- Plot Both FCC and Norway ----------------
trace_sets = {
    "FCC": {"prefix": "log_RL_fcc-test_", "marker": "o"},
    "Norway": {"prefix": "log_RL_norway-test_", "marker": "s"},
}

# Fixed color map
model_color_map = {
    "Pensieve": "blue",
    "Adaptor": "orange"
}

plt.figure()

for trace_label, trace_info in trace_sets.items():
    trace_prefix = trace_info["prefix"]
    trace_marker = trace_info["marker"]

    # Collect Pensieve logs
    pensieve_logs = {
        f: os.path.join(PENSIEVE_DIR, f)
        for f in os.listdir(PENSIEVE_DIR)
        if f.startswith(trace_prefix)
    }

    # Collect Unum logs
    unum_models = {}
    for model in os.listdir(UNUM_DIR):
        if not model.startswith("server_"):
            continue
        model_path = os.path.join(UNUM_DIR, model)
        trace_subdir = os.path.join(model_path, "UDR-3_0_60_40")
        if not os.path.exists(trace_subdir):
            continue

        model_traces = [
            f for f in os.listdir(trace_subdir)
            if f.startswith(trace_prefix)
        ]
        if model_traces:
            unum_models[model] = model_path

    # Find common traces
    common_traces = set(pensieve_logs.keys())
    for model_path in unum_models.values():
        trace_dir = os.path.join(model_path, "UDR-3_0_60_40")
        common_traces &= {
            f for f in os.listdir(trace_dir) if f in pensieve_logs
        }

    # Collect metrics
    model_metrics = {
        f"Pensieve-{trace_label}": []
    }

    for trace_file in common_traces:
        pensieve_avg, pensieve_tail, pensieve_bitrate = compute_metrics(pensieve_logs[trace_file])
        if pensieve_avg is not None:
            model_metrics[f"Pensieve-{trace_label}"].append((pensieve_avg, pensieve_tail, pensieve_bitrate))

        for model, model_path in unum_models.items():
            trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
            if os.path.exists(trace_path):
                rebuffer_ratio, rebuffer_tail, avg_bitrate = compute_metrics(trace_path)
                if rebuffer_ratio is not None and avg_bitrate is not None:
                    server_id = int(model.split('_')[1])
                    model_type = get_model_config(server_id)
                    if model_type == "action_256_20":
                        model_metrics.setdefault(f"Adaptor-{trace_label}", []).append((rebuffer_ratio, rebuffer_tail, avg_bitrate))

    # Compute means and plot
    for model_name, values in model_metrics.items():
        if not values:
            continue
        mean_rebuffer = np.mean([v[0] for v in values])
        mean_tail = np.mean([v[1] for v in values])
        mean_bitrate = np.mean([v[2] for v in values])

        if "Pensieve" in model_name:
            label = f"Pensieve ({trace_label})"
            color = model_color_map["Pensieve"]
        elif "Adaptor" in model_name:
            label = f"Pensieve-Unum-Adaptor ({trace_label})"
            color = model_color_map["Adaptor"]
        else:
            continue

        plt.scatter(mean_rebuffer, mean_bitrate, label=label, color=color,
                    marker=trace_marker, s=120, edgecolors='black', alpha=0.9)

# ----------------- Final Plotting ---------------------
plt.xlabel("Mean 90p Rebuffering Time (ms)", fontsize=18)
plt.ylabel("Mean Bitrate (Kbps)", fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

plot_path = os.path.join(SCATTER_PLOT_DIR, "mean_scatter_fcc_norway_markers.png")
plt.savefig(plot_path)
plt.close()

print(f"Saved mean scatter plot: {plot_path}")
