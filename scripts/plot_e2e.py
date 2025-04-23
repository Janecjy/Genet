import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ----------------- Argument Parser -------------------
parser = argparse.ArgumentParser(description="Generate mean scatter plot from ABR test logs")
parser.add_argument('--filter', choices=['all', 'fcc', 'norway'], default='all',
                    help="Filter traces by 'fcc', 'norway', or 'all'")
args = parser.parse_args()
trace_filter = args.filter

# ------------------ Directories ---------------------
RESULTS_DIR = "/home/jane/Genet/scripts/04_20_model_set"
SCATTER_PLOT_DIR = os.path.join(RESULTS_DIR, "scatter_plots")
os.makedirs(SCATTER_PLOT_DIR, exist_ok=True)

PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/testing_trace_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "04_20_model_set/04_20_model_set")

# ---------------- Configurations ---------------------
adaptor_inputs = ["original_selection", "hidden_state"]
adaptor_hidden_layers = [128, 256]
seeds = [10, 20, 30, 40, 50]

adaptor_configs = [(inp, hl, s) for s in seeds for inp in adaptor_inputs for hl in adaptor_hidden_layers]

def get_model_config(server_id):
    index = (server_id - 1) % len(adaptor_configs)
    adaptor_input, hidden_layer, seed = adaptor_configs[index]
    model_name = f"{'action' if adaptor_input == 'original_selection' else 'hidden'}_{hidden_layer}_{seed}"
    return model_name

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

# -------------- Collect Pensieve Logs ----------------
pensieve_logs = {}
for file in os.listdir(PENSIEVE_DIR):
    if trace_filter == 'fcc' and not file.startswith("log_RL_fcc-test_"):
        continue
    if trace_filter == 'norway' and not file.startswith("log_RL_norway-test_"):
        continue
    if file.startswith("log_RL_"):
        pensieve_logs[file] = os.path.join(PENSIEVE_DIR, file)

# -------------- Collect Unum Models ------------------
unum_models = {}
for model in os.listdir(UNUM_DIR):
    if not model.startswith("server_"):
        continue
    model_path = os.path.join(UNUM_DIR, model)
    trace_subdir = os.path.join(model_path, "UDR-3_0_60_40")

    model_traces = [f for f in os.listdir(trace_subdir) if f.startswith("log_RL_")]
    if trace_filter == 'fcc':
        model_traces = [f for f in model_traces if f.startswith("log_RL_fcc-test_")]
    elif trace_filter == 'norway':
        model_traces = [f for f in model_traces if f.startswith("log_RL_norway-test_")]

    if model_traces:
        unum_models[model] = model_path

# -------------- Find Common Traces -------------------
common_traces = set(pensieve_logs.keys())
for model, model_path in unum_models.items():
    trace_dir = os.path.join(model_path, "UDR-3_0_60_40")
    trace_files = set(os.listdir(trace_dir))
    trace_files = {f for f in trace_files if f in pensieve_logs}
    common_traces &= trace_files

# -------------- Compute Metrics ----------------------
model_metrics = {"pensieve-original": []}
action_models = {}
hidden_models = {}

for trace_file in common_traces:
    pensieve_avg, pensieve_tail, pensieve_bitrate = compute_metrics(pensieve_logs[trace_file])
    if pensieve_avg is not None and pensieve_tail is not None and pensieve_bitrate is not None:
        model_metrics["pensieve-original"].append((pensieve_avg, pensieve_tail, pensieve_bitrate))

    for model, model_path in unum_models.items():
        trace_path = os.path.join(model_path, "UDR-3_0_60_40", trace_file)
        if os.path.exists(trace_path):
            rebuffer_ratio, rebuffer_tail, avg_bitrate = compute_metrics(trace_path)
            if rebuffer_ratio is not None and avg_bitrate is not None:
                server_id = int(model.split('_')[1])
                model_name = get_model_config(server_id)
                model_metrics.setdefault(model_name, []).append((rebuffer_ratio, rebuffer_tail, avg_bitrate))

# -------------- Compute Mean Values ------------------
mean_metrics = {}
for model_name, values in model_metrics.items():
    if values:
        mean_rebuffer = np.mean([v[0] for v in values])
        mean_rebuffer_tail = np.mean([v[1] for v in values])
        mean_bitrate = np.mean([v[2] for v in values])
        mean_metrics[model_name] = (mean_rebuffer, mean_rebuffer_tail, mean_bitrate)
        if model_name.startswith("action"):
            action_models[model_name] = (mean_rebuffer, mean_rebuffer_tail, mean_bitrate)
        elif model_name.startswith("hidden"):
            hidden_models[model_name] = (mean_rebuffer, mean_rebuffer_tail, mean_bitrate)

# ----------------- Plotting --------------------------
plt.figure()

# if "pensieve-original" in mean_metrics:
#     x, y = mean_metrics["pensieve-original"]
#     plt.scatter(x, y, color='blue', label="Pensieve", s=100, edgecolors='black', marker="o")

# for model_name, (x, y) in sorted(action_models.items()):
#     if model_name == "action_256_20":
#         plt.scatter(x, y, color='orange', label="Pensieve-Unum-Adaptor" if "Action" not in plt.gca().get_legend_handles_labels()[1] else "", s=80, alpha=0.8)

for model_name, (avg_rebuf, tail_rebuf, bitrate) in mean_metrics.items():
    if model_name == "pensieve-original":
        plt.scatter(avg_rebuf, bitrate, color='blue', label="Pensieve", s=100, edgecolors='black', marker="o")
        # plt.plot([avg_rebuf, tail_rebuf], [bitrate, bitrate], color='blue', linewidth=3)

for model_name, (avg_rebuf, tail_rebuf, bitrate) in sorted(action_models.items()):
    if model_name == "action_256_20":
        plt.scatter(avg_rebuf, bitrate, color='orange', label="Pensieve-Unum-Adaptor", s=100, alpha=0.8)
        # plt.plot([avg_rebuf, tail_rebuf], [bitrate, bitrate], color='orange', linewidth=3)


# for model_name, (x, y) in sorted(hidden_models.items()):
#     plt.scatter(x, y, color='green', label="Hidden" if "Hidden" not in plt.gca().get_legend_handles_labels()[1] else "", s=80, alpha=0.8)

plt.xlabel("Mean 90p Rebuffering Time (ms)", fontsize=18)
plt.ylabel("Mean Bitrate (Kbps)", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
ax = plt.gca()

# Tick font sizes
ax.tick_params(axis='both', which='major', labelsize=18)

plot_path = os.path.join(SCATTER_PLOT_DIR, f"mean_scatter_{trace_filter}.png")

plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"Saved mean scatter plot: {plot_path}")
print("Mean scatter plot generated successfully.")
