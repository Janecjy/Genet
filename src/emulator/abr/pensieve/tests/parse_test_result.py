import os
import numpy as np
import argparse

def calculate_mean_and_std(rewards):
    """Calculate mean and standard deviation of the reward values."""
    mean = np.mean(rewards)
    std = np.std(rewards)
    total_num = len(rewards)
    return mean, std, total_num

def extract_reward(file_path, skip_header):
    """Extract reward column from a file."""
    rewards = []
    reward_index = -1
    with open(file_path, 'r') as f:
        if not skip_header:
            header = next(f).strip().split()
            reward_index = header.index("reward")
        for line in f:
            values = line.strip().split()
            if values:
                reward = float(values[reward_index])
                rewards.append(reward)
    mean, std, num = calculate_mean_and_std(rewards)
    return mean, std, num

def reward_mean_std(directory, skip_header):
    rewards_means = []
    mean = -1
    std = -1
    total_num = -1
    print(directory)
    print(skip_header)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                mean, std, num = extract_reward(file_path, skip_header)
                rewards_means.append(mean)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    # Calculate mean and standard deviation of the reward values
    if rewards_means:
        mean, std, total_num = calculate_mean_and_std(rewards_means)
        print(f"Average Mean reward: {mean}")
        print(f"Standard deviation of Mean reward: {std}")
        print(f"Total num: {total_num}")
    else:
        print("No reward values found.")

    return mean, std, total_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reward column and calculate mean and std")
    parser.add_argument("directory", help="Directory containing files to process")
    parser.add_argument("--simulation", action="store_true", help="Don't skip header line")
    args = parser.parse_args()

    reward_mean_std(args.directory, args.simulation)
