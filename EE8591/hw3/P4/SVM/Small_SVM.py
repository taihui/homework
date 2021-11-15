############# for small dataset when using SVM
### This is the code for Problem 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


########################## SVM #################################
# calculate precision-recall area under curve
def test_SVM(X, y, model):
    probas_pred = model.predict(X)
    acc = accuracy_score(y, probas_pred)
    return acc

def train_SVM(train_X, train_Y, val_X, val_Y, class_weights):
    print('>>>>>> Start SVM')
    param_grid = [
                 {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5 ],
                  'gamma': [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),1,pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5)],
                  'kernel': ['rbf']},
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




def equal_cost_svm(train_f5, train_f8, val_f5, val_f8, test_f5, test_f8):
    #---load data first
    # train
    train_X_5 = np.load(train_f5)['arr_0'].reshape(-1, 784)
    train_Y_5 = -1*np.ones(len(train_X_5)).reshape(-1,1)
    train_X_8 = np.load(train_f8)['arr_0'].reshape(-1, 784)
    train_Y_8 = np.ones(len(train_X_8)).reshape(-1,1)
    train_X = np.vstack((train_X_5, train_X_8))
    train_Y = np.vstack((train_Y_5, train_Y_8))

    # val
    val_X_5 = np.load(val_f5)['arr_0'].reshape(-1, 784)
    val_Y_5 = -1 * np.ones(len(val_X_5)).reshape(-1, 1)
    val_X_8 = np.load(val_f8)['arr_0'].reshape(-1, 784)
    val_Y_8 = np.ones(len(val_X_8)).reshape(-1, 1)
    val_X = np.vstack((val_X_5, val_X_8))
    val_Y = np.vstack((val_Y_5, val_Y_8))

    # test
    test_X_5 = np.load(test_f5)['arr_0'].reshape(-1, 784)
    test_Y_5 = -1 * np.ones(len(test_X_5)).reshape(-1, 1)
    test_X_8 = np.load(test_f8)['arr_0'].reshape(-1, 784)
    test_Y_8 = np.ones(len(test_X_8)).reshape(-1, 1)
    test_X = np.vstack((test_X_5, test_X_8))
    test_Y = np.vstack((test_Y_5, test_Y_8))

    #---- set up SVM model
    class_weights = 'balanced'
    best_clf, best_ap = train_SVM(train_X, train_Y, val_X, val_Y, class_weights)

    #---- now, let's get its accuray
    train_acc = test_SVM(train_X, train_Y.flatten(), best_clf)
    val_acc = test_SVM(val_X, val_Y.flatten(), best_clf)
    test_acc = test_SVM(test_X, test_Y.flatten(), best_clf)

    print('##################### results for Small dataset of SVM')
    print('')
    print('')
    print('Train_Acc={}'.format(train_acc))
    print('Val_Acc={}'.format(val_acc))
    print('Test_Acc={}'.format(test_acc))
    print('')
    print('')
    print('###############################')


    print('Finished!')


if __name__ == '__main__':
    # ----------- training, validation, and test
    # --- for Equal Weights
    train_f5 = '../Get_Data/small_train_5.npz'
    train_f8 = '../Get_Data/small_train_8.npz'

    val_f5 = '../Get_Data/small_val_5.npz'
    val_f8 = '../Get_Data/small_val_8.npz'

    test_f5 = '../Get_Data/test_5.npz'
    test_f8 = '../Get_Data/test_8.npz'

    equal_cost_svm(train_f5, train_f8, val_f5, val_f8, test_f5, test_f8)