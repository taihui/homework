import torch, os, random, argparse, utils
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import operator

def draw_fig(n_center,final_centers_X,final_centers_Y):
    ############# Get Dataset #############
    train_file = "dataset/hw3/synth.tr"
    test_file = "dataset/hw3/synth.te"
    train_df = pd.read_csv(train_file,sep='\s+')
    train_X = train_df.iloc[:,0:2].to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    train_X_scaled = scaler.transform(train_X)
    train_X_tensor = torch.from_numpy(train_X_scaled).view((-1,2))
    train_Y = train_df.iloc[:,-1].to_numpy()
    train_Y_tensor = torch.from_numpy(train_Y)

    test_df = pd.read_csv(test_file,sep='\s+')
    test_X = test_df.iloc[:,0:2].to_numpy()
    test_X_scaled = scaler.transform(test_X)
    test_X_tensor = torch.from_numpy(test_X_scaled).view((-1,2))
    test_Y = test_df.iloc[:,-1].to_numpy()
    test_Y_tensor = torch.from_numpy(test_Y)



    print("#############")
    cl1_idx = np.where(train_Y==0)[0]
    cl1_X=  train_X[cl1_idx,:]
    cl2_idx = np.where(train_Y==1)[0]
    cl2_X = train_X[cl2_idx,:]
    plt.scatter(cl1_X[:,0], cl1_X[:,1], marker="^", facecolors='none', edgecolors='c')
    plt.scatter(cl2_X[:, 0], cl2_X[:, 1], marker="+",  edgecolors='k')
    plt.scatter(final_centers_X, final_centers_Y, edgecolors='r')

    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    #sorted_zip = sorted(zip(final_centers[:,0], final_centers[:,1]), key=sort_axis)
    #X_denoised_sorted, Y_denoised_sorted = zip(*sorted_zip)
    #plt.plot(X_denoised_sorted,Y_denoised_sorted,color='k')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.savefig('fig/p3_{}.png'.format(str(n_center)))
    plt.show()
    plt.close()

if __name__ == '__main__':
    final_centers_X =  [-0.736623000000000,0.260430000000000,-0.312748000000000,0.485674000000000]
    final_centers_Y = [0.266741000000000,0.356116000000000,0.728615000000000,0.680615000000000]
    draw_fig(4, final_centers_X, final_centers_Y)


    df9 = pd.read_csv('9.csv')
    final_centers_X = df9['x'].to_numpy()
    final_centers_Y = df9['y'].to_numpy()
    draw_fig(9, final_centers_X, final_centers_Y)

    df25 = pd.read_csv('25.csv')
    final_centers_X = df25['x'].to_numpy()
    final_centers_Y = df25['y'].to_numpy()
    draw_fig(25, final_centers_X, final_centers_Y)


