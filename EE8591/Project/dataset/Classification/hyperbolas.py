### This is the code for Problem 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def generate_dateset(num, save_file_name, fig_title, fig_name):
    #------- for negative class
    nx1 = np.random.uniform(low=0.2, high=0.6, size=num)
    nx2 = ((nx1-0.4)*3)**2 + 0.225 + np.random.normal(loc=0.0, scale=0.025, size=num)
    ny = -1*np.ones(num)

    # ------- for positive class
    px1 = np.random.uniform(low=0.4, high=0.8, size=num)
    px2 = 1-((px1 - 0.6) * 3) ** 2 - 0.225 + np.random.normal(loc=0.0, scale=0.025, size=num)
    py = np.ones(num)


    ### visualize the dataset
    plt.figure()
    plt.scatter(nx1, nx2, marker = 'o', c='b', alpha=0.5, label='Negative')
    plt.scatter(px1, px2, marker='x', c='r', alpha=0.5, label='Positive')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(fig_title)
    plt.legend()
    plt.savefig(fig_name, dpi=300)
    plt.close()

    #### now, we want to save our dataset into a csv file
    n_df = pd.DataFrame()
    n_df['x1'] = nx1
    n_df['x2'] = nx2
    n_df['y'] = ny

    p_df = pd.DataFrame()
    p_df['x1'] = px1
    p_df['x2'] = px2
    p_df['y'] = py

    df = pd.concat([n_df, p_df])
    df.to_csv(save_file_name, index=False)



if __name__ == '__main__':
    # #--- for training dataset
    num = 50
    save_file_name = 'Hyperbolas/hyperbolas_train.csv'
    fig_title = 'Training Dataset'
    fig_name = 'Hyperbolas/hyperbolas_train.png'
    generate_dateset(num, save_file_name, fig_title, fig_name)

    # --- for validation dataset
    num = 50
    save_file_name = 'Hyperbolas/hyperbolas_val.csv'
    fig_title = 'Validation Dataset'
    fig_name = 'Hyperbolas/hyperbolas_val.png'
    generate_dateset(num, save_file_name, fig_title, fig_name)

    # --- for test dataset
    num = 1000
    save_file_name = 'Hyperbolas/hyperbolas_test.csv'
    fig_title = 'Test Dataset'
    fig_name = 'Hyperbolas/hyperbolas_test.png'
    generate_dateset(num, save_file_name, fig_title, fig_name)