from parse_test_result import reward_mean_std, extract_reward
import matplotlib.pyplot as plt
import glob

def main():
    model_directories = [("/mydata/results/UDR-3-orig/", False)]
    
    # Automatically collect model directories from /mydata/results/03_17_model_summary/*
    model_subdirs = glob.glob("/mydata/results/03_17_model_summary/*/UDR-3_0_60_40/")
    
    # Extract numbers from model directory names and sort them
    model_subdirs_sorted = sorted(model_subdirs, key=lambda x: int(x.split("server_")[1].split("_nn_model_ep")[0]))
    model_directories.extend([(subdir, False) for subdir in model_subdirs_sorted])
    
    x = ['Pensieve'] + [f'Model {int(subdir.split("server_")[1].split("_nn_model_ep")[0])}' for subdir in model_subdirs_sorted]
    y = []
    stds = []

    plt.figure(figsize=(8, 6))

    for result_path, is_simulation in model_directories:
        print(result_path)
        mean, std, total_num = reward_mean_std(result_path, is_simulation)
        print(mean)
        y.append(mean)
        stds.append(std)
       
    plt.bar(
        x,
        y,
        yerr=stds,
        capsize=5,               # length of the error bar caps
        alpha=0.7,               # transparency for the bars
        width=0.5,
        error_kw=dict(ecolor='black', lw=2)  # style of error bars
    )
    
    plt.ylabel("Mean Reward", fontsize=20)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=18)
    plt.legend(loc='lower right', fontsize=18, frameon=False)

    plt.tight_layout(pad=1.0)
    plt.savefig("mean_rewards_plot.png")

if __name__ == "__main__":
    main()
