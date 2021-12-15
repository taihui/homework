import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



def plot_ripley(input_file, fig_title, fig_name):

    df = pd.read_csv(input_file)
    neg_df = df[df['yc']==0]
    pos_df = df[df['yc']==1]
    nx1 = neg_df['xs'].values
    nx2 = neg_df['ys'].values

    px1 = pos_df['xs'].values
    px2 = pos_df['ys'].values



    ### visualize the dataset
    plt.figure()
    plt.scatter(nx1, nx2, marker='o', c='b', alpha=0.5, label='Negative')
    plt.scatter(px1, px2, marker='x', c='r', alpha=0.5, label='Positive')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(fig_title)
    plt.legend()
    plt.savefig(fig_name, dpi=300)
    plt.close()



if __name__ == '__main__':

    train_file_csv = 'Ripley/Ripley_train.csv'
    fig_name = 'Ripley/Ripley_train.png'
    fig_title = 'Training Dataset'
    plot_ripley(train_file_csv, fig_title, fig_name)

    test_file_csv = 'Ripley/Ripley_test.csv'
    fig_name = 'Ripley/Ripley_test.png'
    fig_title = 'Test Dataset'
    plot_ripley(test_file_csv, fig_title, fig_name)


    pass



