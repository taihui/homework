####################### The script will train our AE model
from nn_bp import *
import os
import glob
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ################ Parameters Settings ######################
    max_epoch = 1001
    print_step = 100
    train_batch_size = 2
    val_batch_size = 2
    test_batch_size = 2

    learning_rate = 1e-4
    opm_tag = "Adam"
    gpu = 0

    train_f5 = '../../../dataset/Classification/MNIST/small_train_5.npz'
    val_f5 = '../../../dataset/Classification/MNIST/small_val_5.npz'
    test_f5 = '../../../dataset/Classification/MNIST/test_5.npz'

    train_f8 = '../../../dataset/Classification/MNIST/small_train_8.npz'
    val_f8 = '../../../dataset/Classification/MNIST/small_val_8.npz'
    test_f8 = '../../../dataset/Classification/MNIST/test_8.npz'

    # train
    train_X_5 = np.load(train_f5)['arr_0'].reshape(-1, 784)
    train_Y_5 = 0 * np.ones(len(train_X_5)).reshape(-1, 1)  ### set digit 5 to lable "0"
    train_X_8 = np.load(train_f8)['arr_0'].reshape(-1, 784)
    train_Y_8 = np.ones(len(train_X_8)).reshape(-1, 1)  ### set digit 8 to lable "1"
    train_X = np.vstack((train_X_5, train_X_8))
    train_Y = np.vstack((train_Y_5, train_Y_8))

    # val
    val_X_5 = np.load(val_f5)['arr_0'].reshape(-1, 784)
    val_Y_5 = 0 * np.ones(len(val_X_5)).reshape(-1, 1)  ### set digit 5 to lable "0"
    val_X_8 = np.load(val_f8)['arr_0'].reshape(-1, 784)
    val_Y_8 = np.ones(len(val_X_8)).reshape(-1, 1)  ### set digit 8 to lable "1"
    val_X = np.vstack((val_X_5, val_X_8))
    val_Y = np.vstack((val_Y_5, val_Y_8))

    # test
    test_X_5 = np.load(test_f5)['arr_0'].reshape(-1, 784)
    test_Y_5 = 0 * np.ones(len(test_X_5)).reshape(-1, 1)  ### set digit 5 to lable "0"
    test_X_8 = np.load(test_f8)['arr_0'].reshape(-1, 784)
    test_Y_8 = np.ones(len(test_X_8)).reshape(-1, 1)  ### set digit 8 to lable "1"
    test_X = np.vstack((test_X_5, test_X_8))
    test_Y = np.vstack((test_Y_5, test_Y_8))
    # for i in range(len(test_Y)):  ### here, we change the label from 0 to -1
    #     if test_Y[i] == 0:
    #         test_Y[i] = -1


    deepreg_train(learning_rate,
                  max_epoch,
                  train_batch_size,
                  val_batch_size,
                  test_batch_size,
                  gpu,
                  print_step,
                  opm_tag,
                  train_X,
                  train_Y,
                  val_X,
                  val_Y,
                  test_X,
                  test_Y)