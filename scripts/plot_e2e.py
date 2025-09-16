"""
Read the log files and plot throughput vs delay.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

# ------------------ Directories ---------------------
RESULTS_DIR = "/home/jane/Genet/scripts/04_20_model_set"
SCATTER_PLOT_DIR = os.path.join(RESULTS_DIR, "scatter_plots")
os.makedirs(SCATTER_PLOT_DIR, exist_ok=True)

PENSIEVE_DIR = os.path.join(RESULTS_DIR, "pensieve-original/testing_trace_mahimahi")
UNUM_DIR = os.path.join(RESULTS_DIR, "04_20_model_set/04_20_model_set")
SAMPLE_DIR = "/home/jane/Genet/abr_trace/testing_trace_mahimahi_sample"

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

# ---------------- Apply Orca figure specifications ----------------
plt.rcParams["font.size"] = 20
plt.figure(figsize=(6, 4))

# Orca color scheme and styling
colors = ["#82B366", "#D79B00", "#9673A6", "#6C8EBF", "#D6B656", "#B85450", "#BF5700", "#FF6347"]
markers = ["o", "P", "^", "s", "p", "d", "v", "X"]
styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (2, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))]

# ---------------- Get sampled traces from plot_reward_boxplot.py ----------------
def get_sampled_traces():
    """Get the same traces used in plot_reward_boxplot.py"""
    sample_traces = {}
    
    for trace_type in ['fcc', 'norway']:
        # Get traces from sample directory (same as plot_reward_boxplot.py)
        type_traces = [
            f for f in os.listdir(SAMPLE_DIR) 
            if f.startswith(f"{trace_type}-test")
        ]
        sample_traces[trace_type] = type_traces
    
    return sample_traces

# ---------------- Plot Both FCC and Norway (using sampled traces) ----------------
sampled_traces = get_sampled_traces()

trace_sets = {
    "FCC": {"traces": sampled_traces.get('fcc', []), "prefix": "log_RL_fcc-test_", "marker": "o"},
    "Norway": {"traces": sampled_traces.get('norway', []), "prefix": "log_RL_norway-test_", "marker": "s"},
}

# Fixed color map using Orca colors
model_color_map = {
    "Pensieve": colors[0],  # "#82B366"
    "Adaptor": colors[1]    # "#D79B00"
}

for trace_label, trace_info in trace_sets.items():
    sample_traces_for_type = trace_info["traces"]
    trace_prefix = trace_info["prefix"]
    trace_marker = trace_info["marker"]
    
    print(f"[{trace_label}] Processing {len(sample_traces_for_type)} sampled traces:")
    for i, trace in enumerate(sample_traces_for_type, 1):
        print(f"  {i}. {trace}")

    # Collect Pensieve logs (only for sampled traces)
    pensieve_logs = {}
    for trace_file in sample_traces_for_type:
        pensieve_trace_name = f"log_RL_{trace_file}"
        pensieve_path = os.path.join(PENSIEVE_DIR, pensieve_trace_name)
        if os.path.exists(pensieve_path):
            pensieve_logs[pensieve_trace_name] = pensieve_path

    # Collect Unum logs
    unum_models = {}
    for model in os.listdir(UNUM_DIR):
        if not model.startswith("server_"):
            continue
        model_path = os.path.join(UNUM_DIR, model)
        trace_subdir = os.path.join(model_path, "UDR-3_0_60_40")
        if not os.path.exists(trace_subdir):
            continue

        # Check if model has any of our sampled traces
        model_traces = [
            f for f in os.listdir(trace_subdir)
            if f in pensieve_logs
        ]
        if model_traces:
            unum_models[model] = model_path

    # Find common traces (intersection of Pensieve and all Unum models, limited to sampled traces)
    common_traces = set(pensieve_logs.keys())
    for model, model_path in unum_models.items():
        trace_dir = os.path.join(model_path, "UDR-3_0_60_40")
        available_traces = set(os.listdir(trace_dir)) & set(pensieve_logs.keys())
        common_traces &= available_traces

    print(f"[{trace_label}] Found {len(common_traces)} common traces between Pensieve and Unum models")

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
            label = f"Pensieve-{trace_label}"
            color = model_color_map["Pensieve"]
        elif "Adaptor" in model_name:
            label = f"Adaptor-{trace_label}"
            color = model_color_map["Adaptor"]
        else:
            continue

        plt.scatter(mean_rebuffer, mean_bitrate, label=label, color=color,
                    marker=trace_marker, s=120, edgecolors='black', alpha=0.9)

# ----------------- Apply Orca styling to final plot ---------------------
plt.xlabel("Mean 90p Rebuffering Time (ms)", fontsize=24)
plt.ylabel("Mean Bitrate (Kbps)", fontsize=24)
plt.grid()

# Use Orca legend styling - positioned inside plot area
plt.legend(loc="center", ncol=1, columnspacing=0.3, 
           handlelength=1.0, handletextpad=0.3, framealpha=0.9)
plt.subplots_adjust(top=0.95, bottom=0.21, left=0.18, right=.98)

plot_path = os.path.join(SCATTER_PLOT_DIR, "thr_delay_output.png")
plot_pdf_path = os.path.join(SCATTER_PLOT_DIR, "thr_delay_output.pdf")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.savefig(plot_pdf_path, dpi=300, bbox_inches='tight')

print(f"Saved plot: {plot_path}")
print(f"Saved PDF: {plot_pdf_path}")
