####################### The script will train our AE model
from nn_bp import *
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    train_file = '../../../dataset/Regression/Sine-squared_50.csv'
    val_file = '../../../dataset/Regression/Sine-squared_100.csv'
    test_file = '../../../dataset/Regression/Sine-squared_500.csv'

    train_df = pd.read_csv(train_file)
    train_X = train_df[['X']].to_numpy()
    train_Y = train_df[['Y']].to_numpy()

    val_df = pd.read_csv(val_file)
    val_X = val_df[['X']].to_numpy()
    val_Y = val_df[['Y']].to_numpy()

    test_df = pd.read_csv(test_file)
    test_X = test_df[['X']].to_numpy()
    test_Y = test_df[['Y']].to_numpy()


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