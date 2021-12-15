####################### The script will train our AE model
from nn_bp import *
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    ################ Parameters Settings ######################
    max_epoch = 10001
    print_step = 100
    train_batch_size = 5
    val_batch_size = 5
    test_batch_size = 5

    learning_rate = 1e-3
    opm_tag = "Adam"
    gpu = 0

    train_file = '../../../dataset/Regression/Life-expectancy.csv'

    train_df = pd.read_csv(train_file)
    train_X = train_df[['MURDER', 'HSGRAD', 'INCOME', 'ILLITERACY']].to_numpy()
    train_Y = train_df[['LIFE EXPECT.']].to_numpy()

    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=35, random_state=42, shuffle=True)
    val_X, test_X, val_Y, test_Y = train_test_split(val_X, val_Y, test_size=20, random_state=42, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X_scaled = scaler.transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)


    deepreg_train(learning_rate,
                  max_epoch,
                  train_batch_size,
                  val_batch_size,
                  test_batch_size,
                  gpu,
                  print_step,
                  opm_tag,
                  train_X_scaled,
                  train_Y,
                  val_X_scaled,
                  val_Y,
                  test_X_scaled,
                  test_Y)