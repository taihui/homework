# This script will use VC Bound and also MSE as a model selection criteria to select the best k value
# for the k-nearest neighbors' regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from dataset import *
import os


def VC_P_Factor(sample_num, k):
    h = sample_num/(k*sample_num**(1/5))
    p = h/sample_num
    risk = p - p*np.log(p) + np.log(sample_num)/(2*sample_num)
    risk = np.sqrt(risk)
    risk = 1-risk
    risk = 1/risk
    risk = np.maximum(0, risk)
    return risk


def Emp_Risk(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


def Process_Dataset_A(sample_num, repeat_num):
    all_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


    result_dict = {'Iter':[], 'Best_MSE_K':[], 'Best_VC_K':[]}

    for cur_repeat in range(repeat_num):
        ### now we need to get our dataset first
        dataset_name = 'Dataset_A_{}.csv'.format(sample_num)
        X, Y, Y_clean = generate_dataset_A(sample_num, dataset_name)

        best_mse = float('inf')
        best_mse_k = 0
        best_vc = float('inf')
        best_vc_k = 0
        ### now we start to explore different K
        for k in all_k:
            neigh = KNeighborsRegressor(n_neighbors=k) # this is our model
            neigh.fit(X, Y) # let's train our model using our dateset
            pred_Y = neigh.predict(X) # let's make prediction using our model

            # now, we want to get the MSE
            cur_mse = Emp_Risk(Y_clean, pred_Y)

            # now, we want to get the VC Penalization factor
            cur_penalty = VC_P_Factor(sample_num, k)

            # now, we want to get the VC bound
            cur_vc_bound = cur_mse * cur_penalty

            # now, we want to save the results under this dataset
            # because finally we will need it to select the best one (which has the lowest error)
            if best_mse > cur_mse:
                best_mse = cur_mse
                best_mse_k = k

            if best_vc > cur_vc_bound:
                best_vc = cur_vc_bound
                best_vc_k = k

        ## after we finish all k, now we get the best k for MSE and VC and we want to save them
        result_dict['Iter'].append(cur_repeat)
        result_dict['Best_MSE_K'].append(best_mse_k)
        result_dict['Best_VC_K'].append(best_vc_k)

    ## after we finish all repeated number, let save our results into csv
    dest_dir = 'Result'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        pass

    dest_file = os.path.join(dest_dir, '{}_dataset_A.csv'.format(sample_num))
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.to_csv(dest_file, index=False)
    print('Congrats! The training results have been saved successfully!')


def Process_Dataset_B(sample_num, repeat_num):
    all_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

    result_dict = {'Iter': [], 'Best_MSE_K': [], 'Best_VC_K': []}

    for cur_repeat in range(repeat_num):
        ### now we need to get our dataset first
        dataset_name = 'Dataset_A_{}.csv'.format(sample_num)
        X, Y, Y_clean = generate_dataset_B(sample_num, dataset_name)

        best_mse = float('inf')
        best_mse_k = 0
        best_vc = float('inf')
        best_vc_k = 0
        ### now we start to explore different K
        for k in all_k:
            neigh = KNeighborsRegressor(n_neighbors=k)  # this is our model
            neigh.fit(X, Y)  # let's train our model using our dateset
            pred_Y = neigh.predict(X)  # let's make prediction using our model

            # now, we want to get the MSE
            cur_mse = Emp_Risk(Y_clean, pred_Y)

            # now, we want to get the VC Penalization factor
            cur_penalty = VC_P_Factor(sample_num, k)

            # now, we want to get the VC bound
            cur_vc_bound = cur_mse * cur_penalty

            # now, we want to save the results under this dataset
            # because finally we will need it to select the best one (which has the lowest error)
            if best_mse > cur_mse:
                best_mse = cur_mse
                best_mse_k = k

            if best_vc > cur_vc_bound:
                best_vc = cur_vc_bound
                best_vc_k = k

        ## after we finish all k, now we get the best k for MSE and VC and we want to save them
        result_dict['Iter'].append(cur_repeat)
        result_dict['Best_MSE_K'].append(best_mse_k)
        result_dict['Best_VC_K'].append(best_vc_k)

    ## after we finish all repeated number, let save our results into csv
    dest_dir = 'Result'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        pass

    dest_file = os.path.join(dest_dir, '{}_dataset_B.csv'.format(sample_num))
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.to_csv(dest_file, index=False)
    print('Congrats! The training results have been saved successfully!')
    print('Finish {}'.format(sample_num))
    print('')


if __name__ == '__main__':
    ### process Dataset A
    repeat_num = 100

    sample_num = 25

    Process_Dataset_A(sample_num,repeat_num)

    sample_num = 50
    Process_Dataset_A(sample_num,repeat_num)

    sample_num = 100
    Process_Dataset_A(sample_num,repeat_num)

    ### process Dataset B
    sample_num = 25
    Process_Dataset_B(sample_num,repeat_num)

    sample_num = 50
    Process_Dataset_B(sample_num,repeat_num)

    sample_num = 100
    Process_Dataset_B(sample_num,repeat_num)
