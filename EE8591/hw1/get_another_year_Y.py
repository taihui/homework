# This script is used to find the empirical distribution of X
# Please not that X(t) = (Z(t)-Z(t-1))/Z(t-1)*100%
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt

def solve_p21 (input_file):
    input_df = pd.read_csv(input_file)
    adjusted_close_org = input_df['Adj Close'].values

    #-----let's get MA first
    total_num = len(adjusted_close_org)
    MA = []
    for i in range(3, total_num):
        z_3 = adjusted_close_org[i]
        z_2 = adjusted_close_org[i-1]
        z_1 = adjusted_close_org[i-2]
        z_0 = adjusted_close_org[i-3]
        cur_ma = (z_3 + z_2 + z_1 + z_0)/4
        MA.append(cur_ma)

    ###### now let's get Y
    Y = []
    total_ma = len(MA)
    for i in range(1, total_ma):
        ma_t = MA[i]
        ma_t_pre = MA[i-1]
        cur_Y = 100*(ma_t - ma_t_pre)/ma_t_pre
        Y.append(cur_Y)

    ###### now, let get its histogram
    bin_num = int((2 * 2.) / (0.2))
    plt.figure()
    plt.hist(Y, bins=bin_num, range=(-2., 2.))
    plt.xlabel('Y values (%)')
    plt.ylabel('Count')
    plt.title('The histogram of Y')
    plt.savefig('hist_Y_2020.png')
    plt.close()

    ###---now, let's get the mean and std of Y
    y_mean = np.mean(Y)
    y_mean_round = round(y_mean, 3)
    y_std = np.std(Y)
    y_std_round = round(y_std, 3)
    result_dict = {'Type': ['Mean (%)', 'Std (%)'],
                   'Value': [y_mean_round, y_std_round]}
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.to_csv('mean_std_Y_2020.csv', index=False)






if __name__ == '__main__':
    # solution to question (a)
    input_file = 'QQQ_2020.csv'
    solve_p21(input_file)