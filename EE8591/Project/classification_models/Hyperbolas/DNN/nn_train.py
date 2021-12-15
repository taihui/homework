####################### The script will train our AE model
from nn_bp import *
import os
import glob
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ################ Parameters Settings ######################
    max_epoch = 5001
    print_step = 100
    train_batch_size = 50
    val_batch_size = 50
    test_batch_size = 50

    learning_rate = 1e-4
    opm_tag = "Adam"
    gpu = 0

    train_file = '../../../dataset/Classification/Hyperbolas/hyperbolas_train.csv'
    val_file = '../../../dataset/Classification/Hyperbolas/hyperbolas_val.csv'
    test_file = '../../../dataset/Classification/Hyperbolas/hyperbolas_test.csv'

    train_df = pd.read_csv(train_file)
    train_X = train_df[['x1', 'x2']].to_numpy()
    train_Y = train_df[['y']].to_numpy()
    train_Y = train_Y.astype('int').flatten()
    for i in range(len(train_Y)):  ### here, we change the label from 0 to -1
        if train_Y[i] == -1:
            train_Y[i] = 0

    val_df = pd.read_csv(val_file)
    val_X = val_df[['x1', 'x2']].to_numpy()
    val_Y = val_df[['y']].to_numpy()
    val_Y = val_Y.astype('int').flatten()
    for i in range(len(val_Y)):  ### here, we change the label from 0 to -1
        if val_Y[i] == -1:
            val_Y[i] = 0


    test_df = pd.read_csv(test_file)
    test_X = test_df[['x1', 'x2']].to_numpy()
    test_Y = test_df[['y']].to_numpy()
    test_Y = test_Y.astype('int').flatten()
    for i in range(len(test_Y)):  ### here, we change the label from 0 to -1
        if test_Y[i] == -1:
            test_Y[i] = 0


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