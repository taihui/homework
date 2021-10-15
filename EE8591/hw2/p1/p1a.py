import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def LSR(data_file):
    input_df = pd.read_csv(data_file)
    input_X = (input_df[['MURDER', 'HSGRAD', 'INCOME', 'ILLITERACY']]).to_numpy()
    input_Y = (input_df[['LIFE EXPECT.']]).to_numpy()

    # first, let's normalize our input features to be the range of [0,1]
    scaler = MinMaxScaler()
    input_X_normalized = scaler.fit_transform(input_X)
    LR_model = LinearRegression().fit(input_X_normalized, input_Y)
    coef = LR_model.coef_
    print('The learnt coefficents are shown below: (the order is) ')
    print("['MURDER', 'HSGRAD', 'INCOME', 'ILLITERACY']")
    print(coef)

    pred_Y = LR_model.predict(input_X_normalized)

    cur_mse = mean_squared_error(input_Y, pred_Y)
    print('The MSE is {}'.format(cur_mse))








if __name__ == '__main__':
    data_file = 'HW2_P1_Data.csv'
    LSR(data_file)
    pass