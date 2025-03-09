from parse_test_result import reward_mean_std
import matplotlib.pyplot as plt

def main():
    model_directories = [("/users/janechen/Genet/src/emulator/abr/pensieve/tests/UDR-3_0_60_40_test_all_trace_summary", False),
                         ("/users/janechen/Genet/fig_reproduce/sigcomm_artifact/synthetic/udr3_result", True)] #(result_path, if_simulation_result)
    x = ['Emulator', 'Simulator']
    y = []

    plt.figure(figsize=(8, 6))

    for result_path, is_simulation in model_directories:
        print(result_path)
        mean, std, total_num = reward_mean_std(result_path, is_simulation)
        print(mean)
        y.append(mean)
    
    plt.bar(x, y)    
    #plt.figure(figsize=(10, 6))
    plt.xlabel("Model Name", fontsize=20)
    plt.ylabel("Mean Reward", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='lower right', fontsize=18, frameon=False)

    plt.tight_layout(pad=1.0)
    plt.savefig("mean_rewards_plot.png")

if __name__ == "__main__":
    main()
