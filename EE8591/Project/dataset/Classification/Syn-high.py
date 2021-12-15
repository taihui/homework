### This is the code for Problem 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def save_code(code, code_file_name):
    np.savez_compressed(code_file_name, code)
    print('The updated code has been saved!')


def generate_dateset(num, save_file_name):
    X = np.random.uniform(low=0.0, high=1.0, size=(num, 20))
    # X_norm = np.linalg.norm(X)
    # X = X / X_norm
    Y = np.zeros(num)
    n_count = 0
    p_count = 0
    for i in range(num):
        cur_sum = X[i, 1] + X[i, 2] + X[i, 3] + X[i, 4] + X[i, 5] + X[i, 6] + X[i, 7] + X[i, 8] + X[i, 9] + X[i, 10]-5
        if cur_sum >= 0:
            Y[i] = 1
            p_count += 1
        else:
            Y[i] = -1
            n_count += 1
    print('N_count={}'.format(n_count))
    print('P_count={}'.format(p_count))

    Y = Y.reshape(-1,1)
    XY = np.hstack((X,Y))
    save_code(XY, save_file_name)


if __name__ == '__main__':
    # #--- for training dataset
    num = 50
    save_file_name = 'Syn-high/syn_high_train.npz'
    generate_dateset(num, save_file_name)

    # --- for validation dataset
    num = 50
    save_file_name = 'Syn-high/syn_high_val.npz'
    generate_dateset(num, save_file_name)

    # --- for test dataset
    num = 1000
    save_file_name = 'Syn-high/syn_high_test.npz'
    generate_dateset(num, save_file_name)