import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def LSR(data_file):
    input_df = pd.read_csv(data_file)
    input_X = (input_df[['MURDER']]).to_numpy()
    input_Y = (input_df[['LIFE EXPECT.']]).to_numpy()

    # first, let's normalize our input features to be the range of [0,1]
    scaler = MinMaxScaler()
    input_X_normalized = scaler.fit_transform(input_X)
    LR_model = LinearRegression().fit(input_X, input_Y)
    pred_Y = LR_model.predict(input_X)

    # Be careful, here we want to order the X values
    sort_idx = np.argsort(input_X, axis=0)
    sorted_X = input_X[sort_idx].flatten()
    sorted_Y = input_Y[sort_idx].flatten()
    sorted_Pred_Y = pred_Y[sort_idx].flatten()


    plt.figure()
    plt.plot(sorted_X, sorted_Y, label = 'Raw data', marker = 'o')
    plt.plot(sorted_X, sorted_Pred_Y, label = 'Learned curve', marker = 's')
    plt.xlabel('MURDER')
    plt.ylabel('LIFE EXPECT.')
    plt.title('MURDER vs. LIFE EXPECT.')
    plt.legend()
    plt.savefig('p1b.png',dpi=300)
    plt.close()
    print('Congrats! The figure has been saved successfully!')




if __name__ == '__main__':
    data_file = 'HW2_P1_Data.csv'
    LSR(data_file)
    pass