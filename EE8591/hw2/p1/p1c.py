import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def get_mse(input_X, input_Y, need_normalization):
    # first, let's normalize our input features to be the range of [0,1]
    if need_normalization:
        scaler = MinMaxScaler()
        input_X_normalized = scaler.fit_transform(input_X)
        LR_model = LinearRegression().fit(input_X_normalized, input_Y)
        pred_Y = LR_model.predict(input_X_normalized)
        cur_mse = mean_squared_error(input_Y, pred_Y)
    else:
        LR_model = LinearRegression().fit(input_X, input_Y)
        pred_Y = LR_model.predict(input_X)
        cur_mse = mean_squared_error(input_Y, pred_Y)

    return cur_mse


def LSR(data_file):
    input_df = pd.read_csv(data_file)
    need_df = input_df[['LIFE EXPECT.', 'MURDER', 'HSGRAD', 'INCOME', 'ILLITERACY']]
    # Using Pearson Correlation
    plt.figure(figsize=(12, 10))
    cor = need_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig('p1c_heatmap.png', dpi=300)
    plt.close()

    # Correlation with output variable
    cor_target = abs(cor["LIFE EXPECT."])
    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0.5]

    print('The left features are:')
    print(relevant_features)
    print('')
    print('')

    print(need_df[["MURDER", "HSGRAD"]].corr())
    print('')
    print('')

    print(need_df[["MURDER", "ILLITERACY"]].corr())
    print('')
    print('')

    print(need_df[["HSGRAD", "ILLITERACY"]].corr())

    #### let's try different subset to see what's the MSE
    # case 1: let's use all features
    input_df = pd.read_csv(data_file)
    input_X = (input_df[['MURDER', 'HSGRAD', 'INCOME', 'ILLITERACY']]).to_numpy()
    input_Y = (input_df[['LIFE EXPECT.']]).to_numpy()
    need_normalization = True
    mse_4_features = get_mse(input_X, input_Y, need_normalization)
    print('')
    print('##########################')
    print('MSE for using 4 features:')
    print(mse_4_features)
    print('##########################')
    print('')

    # case 2: let's use 3 features based their correlations to targets
    input_df = pd.read_csv(data_file)
    input_X = (input_df[['MURDER', 'HSGRAD', 'ILLITERACY']]).to_numpy()
    input_Y = (input_df[['LIFE EXPECT.']]).to_numpy()
    need_normalization = True
    mse_3_features = get_mse(input_X, input_Y, need_normalization)
    print('##########################')
    print('MSE for using 3 features:')
    print(mse_3_features)
    print('##########################')
    print('')

    # case 3: let's use 2 features based their correlations to targets and the correlations between features
    input_df = pd.read_csv(data_file)
    input_X = (input_df[['MURDER', 'HSGRAD']]).to_numpy()
    input_Y = (input_df[['LIFE EXPECT.']]).to_numpy()
    need_normalization = True
    mse_2_features = get_mse(input_X, input_Y, need_normalization)
    print('##########################')
    print('MSE for using 2 features:')
    print(mse_2_features)
    print('##########################')
    print('')

    # case 1: let's use 1 feature1 based their correlations to targets and the correlations between features
    input_df = pd.read_csv(data_file)
    input_X = (input_df[['MURDER']]).to_numpy()
    input_Y = (input_df[['LIFE EXPECT.']]).to_numpy()
    need_normalization = False
    mse_1_features = get_mse(input_X, input_Y, need_normalization)
    print('##########################')
    print('MSE for using 1 features:')
    print(mse_1_features)
    print('##########################')
    print('')


if __name__ == '__main__':
    data_file = 'HW2_P1_Data.csv'
    LSR(data_file)
    pass