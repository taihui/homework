####################### The script will train our AE model
from nn_bp import *
import os
import glob


if __name__ == '__main__':
    ################ Parameters Settings ######################
    max_epoch = 5001
    print_step = 100
    train_batch_size = 100
    val_batch_size = 100
    test_batch_size = 100

    learning_rate = 1e-3
    opm_tag = "SGD"
    gpu = 0

    train_f5 = '../Get_Data/large_train_5.npz'
    train_f8 = '../Get_Data/large_train_8.npz'

    val_f5 = '../Get_Data/large_val_5.npz'
    val_f8 = '../Get_Data/large_val_8.npz'

    test_f5 = '../Get_Data/test_5.npz'
    test_f8 = '../Get_Data/test_8.npz'


    deepreg_train(learning_rate,
                  max_epoch,
                  train_batch_size,
                  val_batch_size,
                  test_batch_size,
                  gpu,
                  print_step,
                  opm_tag,
                  train_f5,
                  train_f8,
                  val_f5,
                  val_f8,
                  test_f5,
                  test_f8
                  )
