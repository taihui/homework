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


def SVM_Clf(train_f5, train_f8, val_f5, val_f8, test_f5, test_f8):
    # ---load data first
    # train
    train_X_5 = np.load(train_f5)['arr_0'].reshape(-1, 784)
    train_Y_5 = -1 * np.ones(len(train_X_5)).reshape(-1, 1) ### set digit 5 to lable "-1"
    train_X_8 = np.load(train_f8)['arr_0'].reshape(-1, 784)
    train_Y_8 = np.ones(len(train_X_8)).reshape(-1, 1) ### set digit 8 to lable "1"
    train_X = np.vstack((train_X_5, train_X_8))
    train_Y = np.vstack((train_Y_5, train_Y_8))

    # val
    val_X_5 = np.load(val_f5)['arr_0'].reshape(-1, 784)
    val_Y_5 = -1 * np.ones(len(val_X_5)).reshape(-1, 1) ### set digit 5 to lable "-1"
    val_X_8 = np.load(val_f8)['arr_0'].reshape(-1, 784)
    val_Y_8 = np.ones(len(val_X_8)).reshape(-1, 1) ### set digit 8 to lable "1"
    val_X = np.vstack((val_X_5, val_X_8))
    val_Y = np.vstack((val_Y_5, val_Y_8))

    # test
    test_X_5 = np.load(test_f5)['arr_0'].reshape(-1, 784)
    test_Y_5 = -1 * np.ones(len(test_X_5)).reshape(-1, 1) ### set digit 5 to lable "-1"
    test_X_8 = np.load(test_f8)['arr_0'].reshape(-1, 784)
    test_Y_8 = np.ones(len(test_X_8)).reshape(-1, 1) ### set digit 8 to lable "1"
    test_X = np.vstack((test_X_5, test_X_8))
    test_Y = np.vstack((test_Y_5, test_Y_8))


    class_weights = 'balanced'
    best_clf, best_acc_train = train_SVM(train_X, train_Y, val_X, val_Y, class_weights)

    # ---- now, let's get its accuracy on test dataset
    test_acc = test_SVM(test_X, test_Y, best_clf)

    #----- get its acc on validation
    val_acc = test_SVM(val_X, val_Y, best_clf)

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
    train_f5 = '../../../dataset/Classification/MNIST/medium_train_5.npz'
    val_f5 = '../../../dataset/Classification/MNIST/medium_val_5.npz'
    test_f5 = '../../../dataset/Classification/MNIST/test_5.npz'

    train_f8 = '../../../dataset/Classification/MNIST/medium_train_8.npz'
    val_f8 = '../../../dataset/Classification/MNIST/medium_val_8.npz'
    test_f8 = '../../../dataset/Classification/MNIST/test_8.npz'

    SVM_Clf(train_f5, train_f8, val_f5, val_f8, test_f5, test_f8)
