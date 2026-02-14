import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Directories ---------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PENSIEVE_DIR = os.path.join(PROJECT_DIR, "results/pensieve")
PENSIEVE_UNUM_DIR = os.path.join(PROJECT_DIR, "results/pensieve-unum")

def compute_mean_reward(log_path):
    """Compute mean reward from a log file, skipping the first row."""
    try:
        df = pd.read_csv(log_path, delim_whitespace=True)
        if 'reward' in df.columns and len(df) > 1:
            return df['reward'][1:].mean()
    except:
        return None
    return None

# ----------------- Collect data ----------------------
all_improvements = []
labels = []

# Get all log files from both directories
pensieve_logs = [f for f in os.listdir(PENSIEVE_DIR) if f.startswith('log_RL_')]
pensieve_unum_logs = [f for f in os.listdir(PENSIEVE_UNUM_DIR) if f.startswith('log_RL_')]

# Find common traces
common_traces = set(pensieve_logs) & set(pensieve_unum_logs)

# print(f"Found {len(common_traces)} common traces between pensieve and pensieve-unum")

# Process each trace type separately
for trace_type in ['fcc', 'norway']:
    trace_prefix = f"log_RL_{trace_type}-test_"
    
    # Filter traces for this type
    type_traces = [t for t in common_traces if t.startswith(trace_prefix)]
    
    if not type_traces:
        print(f"No traces found for {trace_type}")
        continue
    
    # print(f"\n[{trace_type.upper()}] Processing {len(type_traces)} traces")
    
    # Compute improvements for each trace
    improvements = []
    pensieve_rewards = []
    unum_rewards = []
    valid_traces = []
    
    for trace_file in type_traces:
        pensieve_path = os.path.join(PENSIEVE_DIR, trace_file)
        unum_path = os.path.join(PENSIEVE_UNUM_DIR, trace_file)
        
        pensieve_reward = compute_mean_reward(pensieve_path)
        unum_reward = compute_mean_reward(unum_path)
        
        if pensieve_reward is not None and unum_reward is not None and pensieve_reward != 0:
            improvement = (unum_reward - pensieve_reward) / abs(pensieve_reward)
            improvements.append(improvement)
            pensieve_rewards.append(pensieve_reward)
            unum_rewards.append(unum_reward)
            valid_traces.append(trace_file)
    
    if improvements:
        # print(f"[{trace_type.upper()}] Successfully processed {len(improvements)} traces")
        # print(f"[{trace_type.upper()}] Mean Pensieve Reward: {np.mean(pensieve_rewards):.4f}")
        # print(f"[{trace_type.upper()}] Mean Pensieve-UNUM Reward: {np.mean(unum_rewards):.4f}")
        # print(f"[{trace_type.upper()}] Mean Improvement: {np.mean(improvements):.4f} ({np.mean(improvements)*100:.2f}%)")
        all_improvements.append(np.array(improvements))
        labels.append(trace_type.upper())


# ----------------- Plot boxplot ----------------------
if not all_improvements:
    print("No data to plot!")
    exit(1)

plt.figure(figsize=(3, 4))
box = plt.boxplot(
    [imp * 100 for imp in all_improvements],  # convert to %
    labels=labels,
    patch_artist=True,
    widths=0.6,
    boxprops=dict(linewidth=1.5),
    medianprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker='o', markersize=4, linestyle='none')
)

# Print the mean value for each box
for i, improvements in enumerate(all_improvements):
    mean = np.mean(improvements) * 100
    print(f"\nMean improvement for {labels[i]}: {mean:.2f}%")

plt.ylabel("Reward Improvement (%)", fontsize=22, y=0.4)
plt.xticks(fontsize=22, rotation=10)
plt.yticks(fontsize=22)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save plots to results directory
output_dir = os.path.join(PROJECT_DIR, "results")
plot_path = os.path.join(output_dir, "reward_improvement_boxplot.png")
plot_pdf_path = os.path.join(output_dir, "reward_improvement_boxplot.pdf")
plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
plt.savefig(plot_pdf_path, bbox_inches='tight', pad_inches=0.1)

print(f"\nSaved boxplot at: {plot_path}")
print(f"Saved boxplot at: {plot_pdf_path}")

