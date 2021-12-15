import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


########################## SVM #################################
# calculate precision-recall area under curve
def test_SVM(X, y, model):
    probas_pred = model.predict(X)
    acc = accuracy_score(y, probas_pred)
    return acc

def train_SVM(train_X, train_Y, val_X, val_Y, class_weights):
    print('>>>>>> Start SVM')
    param_grid = [{'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
                   'kernel': ['linear']},

                  {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
                   'gamma': [pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2),
                             pow(2, 3), pow(2, 4), pow(2, 5)],
                   'kernel': ['rbf']},
                   
                   {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5],
                   'degree':[1,2,3,4,5],
                   'gamma': [pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2),
                             pow(2, 3), pow(2, 4), pow(2, 5)],
                   'kernel': ['poly']},
                  ]


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


def SVM_Clf(train_file_csv, test_file_csv):
    train_df = pd.read_csv(train_file_csv)
    train_X = train_df[['xs', 'ys']].to_numpy()
    train_Y = train_df[['yc']].to_numpy()
    train_Y = train_Y.astype('int').flatten()
    for i in range(len(train_Y)): ### here, we change the label from 0 to -1
        if train_Y[i]==0:
            train_Y[i] = -1

    test_df = pd.read_csv(test_file_csv)
    test_X = test_df[['xs', 'ys']].to_numpy()
    test_Y = test_df[['yc']].to_numpy()
    test_Y = test_Y.astype('int').flatten()
    for i in range(len(test_Y)): ### here, we change the label from 0 to -1
        if test_Y[i] == 0:
            test_Y[i] = -1

    X_train, X_val, Y_train, Y_val = train_test_split(train_X, train_Y, test_size = 150, random_state = 42, stratify=train_Y, shuffle=True)
    class_weights = 'balanced'
    best_clf, best_acc_train = train_SVM(X_train, Y_train, X_val, Y_val, class_weights)

    # ---- now, let's get its accuracy on test dataset
    test_acc = test_SVM(test_X, test_Y, best_clf)

    #----- get its acc on validation
    val_acc = test_SVM(X_val, Y_val, best_clf)

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
    f.write('For SVM, the training acc is: {}'.format(best_acc_train))
    f.write('#--------------------------------------------\n')
    f.write('#--------------------------------------------\n')
    f.write('\n')
    f.write('\n')
    f.write('For SVM, the validation acc is: {}'.format(val_acc))
    f.write('#--------------------------------------------\n')
    f.write('#--------------------------------------------\n')
    f.write('\n')
    f.write('\n')
    f.write('For SVM, the test acc is: {}'.format(test_acc))
    f.close()


if __name__ == '__main__':
    ### load dataset
    train_file = '../../../dataset/Classification/Ripley/Ripley_train.csv'
    test_file = '../../../dataset/Classification/Ripley/Ripley_test.csv'
    SVM_Clf(train_file, test_file)
