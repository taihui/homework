import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


########################## SVM #################################
# calculate precision-recall area under curve
def test_SVM(X, y, model):
    probas_pred = model.predict(X)
    mse = mean_squared_error(y, probas_pred)
    #acc = accuracy_score(y, probas_pred)
    return mse

def train_SVM(train_X, train_Y, val_X, val_Y):
    print('>>>>>> Start SVM')
    param_grid = [
                  {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
                   'gamma': [pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2),
                             pow(2, 3), pow(2, 4), pow(2, 5)],
                   'kernel': ['rbf']},
                   
                    {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
                   'degree':[1,2,3,4,5],
                   'gamma': [pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2),
                             pow(2, 3), pow(2, 4), pow(2, 5)],
                   'kernel': ['poly']},
                   
                   {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
                   'kernel': ['linear']},
                   
                  ]


    best_ap = float('inf')
    best_clf = SVR()
    for cur_param in ParameterGrid(param_grid):
       cur_clf = SVR(**cur_param)
       cur_clf.fit(train_X, train_Y)
       cur_ap = test_SVM(val_X, val_Y, cur_clf)
       if cur_ap <= best_ap:
           best_ap = cur_ap
           best_clf = cur_clf
    ### after we finish all parameters, use the best parameter to final train our model
    return best_clf, best_ap


def SVM_Clf(train_file_csv):
    train_df = pd.read_csv(train_file_csv)
    train_X = train_df[['MURDER', 'HSGRAD', 'INCOME', 'ILLITERACY']].to_numpy()
    train_Y = train_df[['LIFE EXPECT.']].to_numpy()

    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size = 35, random_state = 42,shuffle=True)
    val_X, test_X, val_Y, test_Y = train_test_split(val_X, val_Y, test_size=20, random_state=42, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X_scaled = scaler.transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)


    best_clf, best_acc_train = train_SVM(train_X_scaled, train_Y, val_X_scaled, val_Y)

    # ---- now, let's get its accuracy on test dataset
    test_acc = test_SVM(test_X_scaled, test_Y, best_clf)

    #----- get its acc on validation
    val_acc = test_SVM(val_X_scaled, val_Y, best_clf)

    ### get the params of the best_clf
    ### save the results into a txt file
    best_params = best_clf.get_params()
    print(best_params)
    f = open('Results.txt', 'a')
    # f.write('For SVM, the best params are:')
    # f.write('\n')
    # f.write(best_params)
    f.write('#--------------------------------------------\n')
    f.write('#--------------------------------------------\n')
    f.write('\n')
    f.write('\n')
    f.write('For SVM, the training MSE is: {}'.format(best_acc_train))
    f.write('#--------------------------------------------\n')
    f.write('#--------------------------------------------\n')
    f.write('\n')
    f.write('\n')
    f.write('For SVM, the validation MSE is: {}'.format(val_acc))
    f.write('#--------------------------------------------\n')
    f.write('#--------------------------------------------\n')
    f.write('\n')
    f.write('\n')
    f.write('For SVM, the test MSE is: {}'.format(test_acc))
    f.close()


if __name__ == '__main__':
    ### load dataset
    train_file = '../../../dataset/Regression/Life-expectancy.csv'
    # test_file = '../../../dataset/Regression/Ripley/Ripley_test.csv'
    SVM_Clf(train_file)
