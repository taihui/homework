import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



def txt2csv():
    train_file_raw = 'P3_results/Ripley/SYNTH.TR'
    train_file_csv = 'P3_results/Ripley/Ripley_train.csv'

    test_file_raw = 'P3_results/Ripley/SYNTH.TE'
    test_file_csv = 'P3_results/Ripley/Ripley_test.csv'

    train_df = pd.read_csv(train_file_raw, sep='\s+')
    train_df.to_csv(train_file_csv, index=False)

    test_df = pd.read_csv(test_file_raw, sep='\s+')
    test_df.to_csv(test_file_csv, index=False)


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
                   'degree':[1,2,3,4,5,6,7,8,9,10],
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


def SVM_Clf(fig_name):
    train_file_csv = 'P3_results/Ripley/Ripley_train.csv'
    train_df = pd.read_csv(train_file_csv)
    train_X = train_df[['xs', 'ys']].to_numpy()
    train_Y = train_df[['yc']].to_numpy()
    train_Y = train_Y.astype('int').flatten()

    test_file_csv = 'P3_results/Ripley/Ripley_test.csv'
    test_df = pd.read_csv(test_file_csv)
    test_X = test_df[['xs', 'ys']].to_numpy()
    test_Y = test_df[['yc']].to_numpy()
    test_Y = test_Y.astype('int').flatten()

    X_train, X_val, Y_train, Y_val = train_test_split(train_X, train_Y, test_size = 0.2, random_state = 42, stratify=train_Y, shuffle=True)
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


    # ---- now, let's visualize it
    # plot the samples
    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y, cmap=plt.cm.Paired, edgecolors="k")

    # plot the decision functions for both classifiers
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # get the separating hyperplane
    Z = best_clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    a = ax.contour(XX, YY, Z, colors="r", levels=[0], linestyles=["-"])
    plt.legend(
        [a.collections[0]],
        ["Boundary"],
        loc="upper right",
    )


    plt.title('SVM on Ripley Dataset: Test Accuracy = {}'.format(test_acc))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(fig_name, dpi=300)
    plt.close()

    # #---- now, let's visualize its prediction
    # plot_decision_regions(test_X, test_Y, clf=best_clf, legend=2)

    print('Finished!')

#------------------------- for KNN-----------------------------#
def test_KNN(X, y, model):
    probas_pred = model.predict(X)
    acc = accuracy_score(y, probas_pred)
    return acc

def train_KNN(train_X, train_Y, val_X, val_Y):
    print('>>>>>> Start KNN')
    param_grid = [
                  {'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51],}
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






def KNN_Clf(fig_name):
    train_file_csv = 'P3_results/Ripley/Ripley_train.csv'
    train_df = pd.read_csv(train_file_csv)
    train_X = train_df[['xs', 'ys']].to_numpy()
    train_Y = train_df[['yc']].to_numpy()
    train_Y = train_Y.astype('int').flatten()

    test_file_csv = 'P3_results/Ripley/Ripley_test.csv'
    test_df = pd.read_csv(test_file_csv)
    test_X = test_df[['xs', 'ys']].to_numpy()
    test_Y = test_df[['yc']].to_numpy()
    test_Y = test_Y.astype('int').flatten()

    X_train, X_val, Y_train, Y_val = train_test_split(train_X, train_Y, test_size = 0.2, random_state = 42, stratify=train_Y, shuffle=True)
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


    # ---- now, let's visualize it
    # plot the samples
    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_Y, cmap=plt.cm.Paired, edgecolors="k")

    # plot the decision functions for both classifiers
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # get the separating hyperplane
    Z = best_clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    a = ax.contour(XX, YY, Z, colors="r", levels=[0], linestyles=["-"])

    plt.legend(
        [a.collections[0]],
        ["Boundary"],
        loc="upper right",
    )

    plt.title('KNN on Ripley Dataset: Test Accuracy = {}'.format(test_acc))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(fig_name, dpi=300)
    plt.close()

    # #---- now, let's visualize its prediction
    # plot_decision_regions(test_X, test_Y, clf=best_clf, legend=2)

    print('Finished!')














if __name__ == '__main__':
    # txt2csv()
    fig_name = 'P3_results/SVM_Ripley.png'
    SVM_Clf(fig_name)

    fig_name = 'P3_results/KNN_Ripley.png'
    KNN_Clf(fig_name)

    pass



