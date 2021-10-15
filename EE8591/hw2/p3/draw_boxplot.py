# This script will draw the boxplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_boxplot(input_file_1, input_file_2, input_file_3, save_mse_fig, save_vc_fig):
    df_1 = pd.read_csv(input_file_1)
    df_2 = pd.read_csv(input_file_2)
    df_3 = pd.read_csv(input_file_3)

    #process MSE
    Best_MSE_K_1 = df_1['Best_MSE_K'].values
    Best_MSE_K_2 = df_2['Best_MSE_K'].values
    Best_MSE_K_3 = df_3['Best_MSE_K'].values

    Best_MSE_K = [Best_MSE_K_1, Best_MSE_K_2, Best_MSE_K_3]

    plt.figure()
    plt.boxplot(Best_MSE_K)
    x = [1,2,3]
    plt.xticks(x, ('25', '50', '100'))
    plt.xlabel('Sample Size')
    plt.ylabel('Optimal k')
    plt.title('MSE (measurement)')
    # show plot
    plt.savefig(save_mse_fig, dpi=300)
    plt.close()

    # process VC
    Best_VC_K_1 = df_1['Best_VC_K'].values
    Best_VC_K_2 = df_2['Best_VC_K'].values
    Best_VC_K_3 = df_3['Best_VC_K'].values

    Best_VC_K = [Best_VC_K_1, Best_VC_K_2, Best_VC_K_3]

    plt.figure()
    plt.boxplot(Best_VC_K)
    x = [1, 2, 3]
    plt.xticks(x, ('25', '50', '100'))
    plt.xlabel('Sample Size')
    plt.ylabel('Optimal k')
    plt.title('VC Bound (measurement)')
    # show plot
    plt.savefig(save_vc_fig, dpi=300)
    plt.close()









if __name__ == '__main__':
    #for dataset A
    dataset_name = 'A'
    input_file_1 = 'Result/25_dataset_{}.csv'.format(dataset_name)
    input_file_2 = 'Result/50_dataset_{}.csv'.format(dataset_name)
    input_file_3 = 'Result/100_dataset_{}.csv'.format(dataset_name)
    save_mse_fig = 'Result/{}_MSE.png'.format(dataset_name)
    save_vc_fig = 'Result/{}_VC.png'.format(dataset_name)
    draw_boxplot(input_file_1, input_file_2, input_file_3, save_mse_fig, save_vc_fig)

    # for dataset B
    dataset_name = 'B'
    input_file_1 = 'Result/25_dataset_{}.csv'.format(dataset_name)
    input_file_2 = 'Result/50_dataset_{}.csv'.format(dataset_name)
    input_file_3 = 'Result/100_dataset_{}.csv'.format(dataset_name)
    save_mse_fig = 'Result/{}_MSE.png'.format(dataset_name)
    save_vc_fig = 'Result/{}_VC.png'.format(dataset_name)
    draw_boxplot(input_file_1, input_file_2, input_file_3, save_mse_fig, save_vc_fig)