import os
import numpy as np
import matplotlib.pyplot as plt

def extract_reward(file_path, skip_header):
    """Extract reward column from a file."""
    rewards = []
    ts = []
    reward_index = -1
    with open(file_path, 'r') as f:
        if not skip_header:
            header = next(f).strip().split()
            reward_index = header.index("reward")
            # print("reward index: ", reward_index)
        for line in f:
            values = line.strip().split()
            if values:
                reward = float(values[reward_index])
                rewards.append(reward)
                ts.append(float(values[0]))
    return ts, rewards

def clean_ts(ts):
    """Convert timestamps to relative time (starting at 0)."""
    return [t - ts[0] for t in ts] if ts else []

# Define model directories
model_directories = {
    "UDR-3-orig": "/mydata/results/UDR-3-orig/",
    "Pensieve-10": "/mydata/results/03_12_model_summary_subset/server_1_nn_model_ep_280/UDR-3_0_60_40/",
    "Pensieve-20": "/mydata/results/03_12_model_summary_subset/server_2_nn_model_ep_280/UDR-3_0_60_40/",
    "Pensieve-30": "/mydata/results/03_12_model_summary_subset/server_3_nn_model_ep_280/UDR-3_0_60_40/",
    "Pensieve-40": "/mydata/results/03_12_model_summary_subset/server_4_nn_model_ep_280/UDR-3_0_60_40/",
    "Pensieve-50": "/mydata/results/03_12_model_summary_subset/server_5_nn_model_ep_280/UDR-3_0_60_40/",
    "Pensieve-Hidden": "/mydata/results/03_12_model_summary_subset/server_10_nn_model_ep_130/UDR-3_0_60_40/",
}

# Step 1: Extract per-trace mean rewards for each model
mean_rewards = {model: {} for model in model_directories}

for model_name, model_dir in model_directories.items():
    print(f"Processing {model_name}...")
    for trace_file in os.listdir(model_dir):
        trace_path = os.path.join(model_dir, trace_file)
        if os.path.isfile(trace_path):
            ts, rewards = extract_reward(trace_path, False)
            mean_rewards[model_name][trace_file] = rewards

# Step 2: Find the trace with the highest improvement
max_improvement = float('-inf')
best_trace = None
best_model = None

for trace in mean_rewards["UDR-3-orig"]:
    orig_rewards = mean_rewards["UDR-3-orig"].get(trace, None)
    if orig_rewards is None:
        continue

    for model_name in mean_rewards:
        if model_name == "UDR-3-orig":
            continue  # Skip baseline

        if trace in mean_rewards[model_name]:
            improvement = np.mean(mean_rewards[model_name][trace]) - np.mean(orig_rewards)
            if improvement > max_improvement:
                max_improvement = improvement
                best_trace = trace
                best_model = model_name

if best_trace and best_model:
    print(f"\n=== Best Improvement Found ===")
    print(f"Trace: {best_trace}")
    print(f"Max Improvement: {max_improvement:.3f}")
    print(f"Best Model: {best_model}")
    # Extract rewards for both models
    ts_original, original_rewards = extract_reward(os.path.join(model_directories["UDR-3-orig"], best_trace), False)
    ts_best, best_model_rewards = extract_reward(os.path.join(model_directories[best_model], best_trace), False)

    # Define Colors and Styles
    colors_dict = {
        "UDR-3-orig": "#6C8EBF",
        "Pensieve-10": "#B85450",
        "Pensieve-20": "#D79B00",
        "Pensieve-30": "#B85450",
        "Pensieve-40": "#D6B656",
        "Pensieve-50": "#9673A6",
        "Pensieve-Hidden": "#6C8EBF",
    }

    styles_dict = {
        "UDR-3-orig": "-",
        "Pensieve-10": "-",
        "Pensieve-20": "-.",
        "Pensieve-30": ":",
        "Pensieve-40": (0, (3, 1, 1, 1)),
        "Pensieve-50": (0, (2, 1)),
        "Pensieve-Hidden": (0, (5, 1)),
    }

    # Plot settings
    plt.figure(figsize=(10, 3.6))
    plt.rcParams["font.size"] = 16

    # Plot original model
    plt.plot(clean_ts(ts_original[1:]), original_rewards[1:], label="Pensieve", 
             color=colors_dict["UDR-3-orig"], linestyle=styles_dict["UDR-3-orig"], linewidth=4)

    # Plot best model
    plt.plot(clean_ts(ts_best[1:]), best_model_rewards[1:], label="Pensieve-Unum", 
             color=colors_dict[best_model], linestyle=styles_dict[best_model], linewidth=4)

    print("Mean reward for UDR-3-orig: ", np.mean(original_rewards[1:]))
    print("Mean reward for Pensieve-Unum: ", np.mean(best_model_rewards[1:]))
    print("Improvement rate: ", (np.mean(best_model_rewards[1:]) - np.mean(original_rewards[1:]))/np.mean(original_rewards[1:]))

    # Formatting
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    # plt.legend(loc="lower right", fontsize=14)
    plt.xlim([0, 30])
    # plt.tight_layout(pad=1.0)
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.title(f"Reward Timeline: {best_trace} ({best_model} vs. UDR-3-orig)", fontsize=16)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, fontsize=20, frameon=False)
    plt.subplots_adjust(top=0.85, bottom=0.18, left=0.1, right=0.98)
    
    plt.savefig("reward_comparison.png")
    print("Saved plot as reward_comparison.png")

else:
    print("No valid comparisons found.")
