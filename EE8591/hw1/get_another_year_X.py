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
    # adjusted_close_forward = np.insert(adjusted_close_org, 0, 0)[0:-1]
    # numerator = (adjusted_close_org - adjusted_close_forward)[1:]
    # denominator = adjusted_close_forward.copy()[1:]
    # X = (numerator/denominator) *100
    X = []
    total_num = len(adjusted_close_org)
    for i in range(1, total_num):
        z_t = adjusted_close_org[i]
        z_t_pre = adjusted_close_org[i-1]
        cur_X = 100*(z_t - z_t_pre)/z_t_pre
        X.append(cur_X)
    bin_num = int((2*2.)/(0.2))
    plt.figure()
    plt.hist(X, bins=bin_num, range=(-2., 2.))
    plt.xlabel('X values (%)')
    plt.ylabel('Count')
    plt.title('The histogram of X')
    plt.savefig('hist_X_2020.png')
    plt.close()

    ###---now, let's get the mean and std of X
    x_mean = np.mean(X)
    x_mean_round = round(x_mean,3)
    x_std = np.std(X)
    x_std_round = round(x_std,3)
    result_dict= {'Type':['Mean (%)', 'Std (%)'],
                  'Value':[x_mean_round, x_std_round]}
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.to_csv('mean_std_X_2020.csv', index=False)




    ###---now, approximate the X using the normal distribution
    mean = x_mean
    standard_deviation = x_std

    x_values = np.arange(-2, 2, 0.01)
    y_values = scipy.stats.norm(mean, standard_deviation)

    plt.figure()
    plt.plot(x_values, y_values.pdf(x_values))
    plt.xlabel('X values (%)')
    plt.ylabel('Frequency')
    plt.title('The PDF of X under normal distribution')
    plt.xlim([-2.1, 2.1])
    plt.savefig('pdf_X_2020.png')
    plt.close()


if __name__ == '__main__':
    # solution to question (a)
    input_file = 'QQQ_2020.csv'
    solve_p21(input_file)