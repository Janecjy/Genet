from parse_test_result import reward_mean_std, extract_reward
import matplotlib.pyplot as plt

def main():
    model_directories = [("/mydata/results/UDR-3-orig/", False),
                         ("/mydata/results/03_12_model_summary_subset/server_1_nn_model_ep_280/UDR-3_0_60_40/", False)]#,
                        #  ("/mydata/results/03_12_model_summary_subset/server_2_nn_model_ep_280/UDR-3_0_60_40/", False),
                        #  ("/mydata/results/03_12_model_summary_subset/server_3_nn_model_ep_280/UDR-3_0_60_40/", False),
                        #  ("/mydata/results/03_12_model_summary_subset/server_4_nn_model_ep_280/UDR-3_0_60_40/", False),
                        #  ("/mydata/results/03_12_model_summary_subset/server_5_nn_model_ep_280/UDR-3_0_60_40/", False),
                        #  ("/mydata/results/03_12_model_summary_subset/server_10_nn_model_ep_130/UDR-3_0_60_40/", False)] #(result_path, if_simulation_result)
    x = ['Pensieve', 'Pensieve-Unum-Action-Adaptor']#, 'Pensieve-20', 'Pensieve-30', 'Pensieve-40', 'Pensieve-50', 'Pensieve-Hidden']
    y = []

    plt.figure(figsize=(8, 6))

    for result_path, is_simulation in model_directories:
        print(result_path)
        mean, std, total_num = reward_mean_std(result_path, is_simulation)
        print(mean)
        y.append(mean)
       
    plt.bar(
        x,
        y,
        yerr=std,
        capsize=5,               # length of the error bar caps
        alpha=0.7,               # transparency for the bars
        width=0.5,
        error_kw=dict(ecolor='black', lw=2)  # style of error bars
    )
    #plt.figure(figsize=(10, 6))
    # plt.xlabel("Model Name", fontsize=20)
    plt.ylabel("Mean Reward", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='lower right', fontsize=18, frameon=False)

    plt.tight_layout(pad=1.0)
    plt.savefig("mean_rewards_plot.png")

if __name__ == "__main__":
    main()
