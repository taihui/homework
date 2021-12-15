import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    train_file_csv = 'Ripley/Ripley_train.csv'
    train_df = pd.read_csv(train_file_csv)
    # X = train_df[['xs','ys']].to_numpy()
    # Y = train_df['yc'].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(train_df[['xs','ys']], train_df['yc'], test_size = 150, random_state = 42, shuffle=True, stratify=train_df['yc'])

    final_trian_X_file = 'Ripley/Final_Ripley_train_X.csv'
    X_train.to_csv(final_trian_X_file, index=False)

    final_trian_Y_file = 'Ripley/Final_Ripley_train_Y.csv'
    y_train.to_csv(final_trian_Y_file, index=False)

    final_trian_X_file = 'Ripley/Final_Ripley_val_X.csv'
    X_val.to_csv(final_trian_X_file, index=False)

    final_trian_Y_file = 'Ripley/Final_Ripley_val_Y.csv'
    y_val.to_csv(final_trian_Y_file, index=False)


