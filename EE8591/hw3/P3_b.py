import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



def generate_dataset(num, save_file_name):
    X = np.random.uniform(low=0.0, high=1.0, size=(num,20))
    # X_norm = np.linalg.norm(X)
    # X = X / X_norm
    Y = np.zeros(num)
    n_count = 0
    p_count = 0
    for i in range(num):
        cur_sum = X[i,1]+X[i,3]+X[i,5]+X[i,7]+X[i,9]+X[i,11]+X[i,13]+X[i,15]+X[i,17]+X[i,19]
        if cur_sum>5:
            Y[i] = 1
            p_count += 1
        else:
            Y[i] = -1
            n_count += 1
    print('N_count={}'.format(n_count))
    print('P_count={}'.format(p_count))

    df = pd.DataFrame()
    df['xs'] = X[:,0].flatten()
    df['ys'] = X[:,1].flatten()
    df['yc'] = Y.flatten()
    df.to_csv(save_file_name, index=False)







########################## SVM #################################
# calculate precision-recall area under curve
def test_SVM(X, y, model):
    probas_pred = model.predict(X)
    acc = accuracy_score(y, probas_pred)
    return acc

def train_SVM(train_X, train_Y, val_X, val_Y, class_weights):
    print('>>>>>> Start SVM')
    param_grid = [
        {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
         'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         'gamma': [pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2),
                   pow(2, 3), pow(2, 4), pow(2, 5)],
         'kernel': ['poly']},
    ]
    # param_grid = [
    #     {'C': [1e-1, 1,],
    #      'gamma': [pow(2, 1)],
    #      'kernel': ['rbf']},
    # ]

    best_ap = 0
    best_clf = SVC(class_weight=class_weights)
    for cur_param in ParameterGrid(param_grid):
       cur_clf = SVC(**cur_param, class_weight=class_weights)
       cur_clf.fit(train_X, train_Y)
       cur_ap = test_SVM(val_X, val_Y, cur_clf)
       if cur_ap >= best_ap:
           best_ap = cur_ap
           best_clf = cur_clf
    ### after we finish all parameters, use the best parameter to final train our model
    return best_clf, best_ap


def SVM_Clf():
    train_file_csv = 'P3_results/High_Dim/HD_train.csv'
    train_df = pd.read_csv(train_file_csv)
    X_train = train_df[['xs', 'ys']].to_numpy()
    Y_train = train_df[['yc']].to_numpy()
    Y_train = Y_train.astype('int').flatten()

    val_file_csv = 'P3_results/High_Dim/HD_val.csv'
    val_df = pd.read_csv(val_file_csv)
    X_val = val_df[['xs', 'ys']].to_numpy()
    Y_val = val_df[['yc']].to_numpy()
    Y_val = Y_val.astype('int').flatten()

    test_file_csv = 'P3_results/High_Dim/HD_test.csv'
    test_df = pd.read_csv(test_file_csv)
    test_X = test_df[['xs', 'ys']].to_numpy()
    test_Y = test_df[['yc']].to_numpy()
    test_Y = test_Y.astype('int').flatten()


    class_weights = 'balanced'
    best_clf, best_acc_train = train_SVM(X_train, Y_train, X_val, Y_val, class_weights)

    # ---- now, let's get its accuracy on test dataset
    test_acc = test_SVM(test_X, test_Y, best_clf)

    ### get the params of the best_clf
    print('#---------------------------------')
    print('#---------------------------------')
    best_params = best_clf.get_params()
    print('For SVM, the best params are:')
    print(best_params)
    print('For SVM, the training error is: {}'.format(best_acc_train))
    print('For SVM, the test error is: {}'.format(test_acc))
    print('#---------------------------------')
    print('#---------------------------------')

    print('Finished!')

#------------------------- for KNN-----------------------------#
def test_KNN(X, y, model):
    probas_pred = model.predict(X)
    acc = accuracy_score(y, probas_pred)
    return acc

def train_KNN(train_X, train_Y, val_X, val_Y):
    print('>>>>>> Start KNN')
    param_grid = [
        {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,], }
    ]

    # param_grid = [
    #     {'C': [1e-1, 1,],
    #      'gamma': [pow(2, 1)],
    #      'kernel': ['rbf']},
    # ]

    best_ap = 0
    best_clf = KNeighborsClassifier()
    for cur_param in ParameterGrid(param_grid):
       cur_clf = KNeighborsClassifier(**cur_param)
       cur_clf.fit(train_X, train_Y)
       cur_ap = test_KNN(val_X, val_Y, cur_clf)
       if cur_ap >= best_ap:
           best_ap = cur_ap
           best_clf = cur_clf
    ### after we finish all parameters, use the best parameter to final train our model
    return best_clf, best_ap






def KNN_Clf():
    train_file_csv = 'P3_results/High_Dim/HD_train.csv'
    train_df = pd.read_csv(train_file_csv)
    X_train = train_df[['xs', 'ys']].to_numpy()
    Y_train = train_df[['yc']].to_numpy()
    Y_train = Y_train.astype('int').flatten()

    val_file_csv = 'P3_results/High_Dim/HD_val.csv'
    val_df = pd.read_csv(val_file_csv)
    X_val = val_df[['xs', 'ys']].to_numpy()
    Y_val = val_df[['yc']].to_numpy()
    Y_val = Y_val.astype('int').flatten()

    test_file_csv = 'P3_results/High_Dim/HD_test.csv'
    test_df = pd.read_csv(test_file_csv)
    test_X = test_df[['xs', 'ys']].to_numpy()
    test_Y = test_df[['yc']].to_numpy()
    test_Y = test_Y.astype('int').flatten()


    best_clf, best_acc_train = train_KNN(X_train, Y_train, X_val, Y_val)

    # ---- now, let's get its accuracy on test dataset
    test_acc = test_KNN(test_X, test_Y, best_clf)

    ### get the params of the best_clf
    print('#---------------------------------')
    print('#---------------------------------')
    best_params = best_clf.get_params()
    print('For KNN, the best params are:')
    print(best_params)
    print('For KNN, the training error is: {}'.format(best_acc_train))
    print('For KNN, the test error is: {}'.format(test_acc))
    print('#---------------------------------')
    print('#---------------------------------')

    print('Finished!')














if __name__ == '__main__':
    #----- generate dataset
    # num = 50
    # save_file_name = 'P3_results/High_Dim/HD_train.csv'
    # generate_dataset(num, save_file_name)
    #
    # num = 50
    # save_file_name = 'P3_results/High_Dim/HD_val.csv'
    # generate_dataset(num, save_file_name)
    #
    # num = 1000
    # save_file_name = 'P3_results/High_Dim/HD_test.csv'
    # generate_dataset(num, save_file_name)



    SVM_Clf()


    KNN_Clf()

    pass



