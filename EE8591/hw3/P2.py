### This is the code for Problem 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



def generate_dateset(num, save_file_name, fig_title, fig_name):
    #------- for negative class
    nx1 = np.random.uniform(low=0.2, high=0.6, size=num)
    nx2 = ((nx1-0.4)*3)**2 + 0.225 + np.random.normal(loc=0.0, scale=0.025, size=num)
    ny = -1*np.ones(num)

    # ------- for positive class
    px1 = np.random.uniform(low=0.4, high=0.8, size=num)
    px2 = 1-((px1 - 0.6) * 3) ** 2 - 0.225 + np.random.normal(loc=0.0, scale=0.025, size=num)
    py = np.ones(num)


    ### visualize the dataset
    plt.figure()
    plt.scatter(nx1, nx2, marker = 'o', c='b', alpha=0.5, label='Negative')
    plt.scatter(px1, px2, marker='x', c='r', alpha=0.5, label='Positive')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(fig_title)
    plt.legend()
    plt.savefig(fig_name, dpi=300)
    plt.close()

    #### now, we want to save our dataset into a csv file
    n_df = pd.DataFrame()
    n_df['x1'] = nx1
    n_df['x2'] = nx2
    n_df['y'] = ny

    p_df = pd.DataFrame()
    p_df['x1'] = px1
    p_df['x2'] = px2
    p_df['y'] = py

    df = pd.concat([n_df, p_df])
    df.to_csv(save_file_name, index=False)


############ visualization ###################
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out






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




def equal_cost_svm(train_file, val_file, test_file, fig_name):
    #---load data first
    train_df = pd.read_csv(train_file)
    train_X = train_df[['x1', 'x2']].to_numpy()
    train_Y = train_df[['y']].to_numpy()
    train_Y = train_Y.astype('int').flatten()

    val_df = pd.read_csv(val_file)
    val_X = val_df[['x1', 'x2']].to_numpy()
    val_Y = val_df[['y']].to_numpy()
    val_Y = val_Y.astype('int').flatten()

    test_df = pd.read_csv(test_file)
    test_X = test_df[['x1', 'x2']].to_numpy()
    test_Y = test_df[['y']].to_numpy()
    test_Y = test_Y.astype('int').flatten()

    #---- set up SVM model
    class_weights = 'balanced'
    best_clf, best_ap = train_SVM(train_X, train_Y, val_X, val_Y, class_weights)

    #---- now, let's get its accuracy
    test_acc = test_SVM(test_X, test_Y, best_clf)


    #---- now, let's visualize it
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
    a = ax.contour(XX, YY, Z, colors="b", levels=[0], linestyles=["-"])

    plt.legend(
        [a.collections[0]],
        ["Equal Costs"],
        loc="upper right",
    )

    plt.title('Equal Costs: Test Accuracy = {}'.format(test_acc))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(fig_name, dpi=300)
    plt.close()

    # #---- now, let's visualize its prediction
    # plot_decision_regions(test_X, test_Y, clf=best_clf, legend=2)

    print('Finished!')


def unequal_cost_svm(train_file, val_file, test_file, fig_name):
    # ---load data first
    train_df = pd.read_csv(train_file)
    train_X = train_df[['x1', 'x2']].to_numpy()
    train_Y = train_df[['y']].to_numpy()
    train_Y = train_Y.astype('int').flatten()

    val_df = pd.read_csv(val_file)
    val_X = val_df[['x1', 'x2']].to_numpy()
    val_Y = val_df[['y']].to_numpy()
    val_Y = val_Y.astype('int').flatten()

    test_df = pd.read_csv(test_file)
    test_X = test_df[['x1', 'x2']].to_numpy()
    test_Y = test_df[['y']].to_numpy()
    test_Y = test_Y.astype('int').flatten()

    # ---- set up SVM model
    class_weights = {-1:2, 1:1}
    best_clf, best_ap = train_SVM(train_X, train_Y, val_X, val_Y, class_weights)

    # ---- now, let's get its accuracy
    test_acc = test_SVM(test_X, test_Y, best_clf)

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
        ["Unequal Costs"],
        loc="upper right",
    )

    plt.title('Unequal Costs: Test Accuracy = {}'.format(test_acc))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(fig_name, dpi=300)
    plt.close()

    # #---- now, let's visualize its prediction
    # plot_decision_regions(test_X, test_Y, clf=best_clf, legend=2)

    print('Finished!')


if __name__ == '__main__':
    # #--- for training dataset
    # num = 50
    # save_file_name = 'results/P2_train.csv'
    # fig_title = 'Training Dataset'
    # fig_name = 'results/P2_train.png'
    # generate_dateset(num, save_file_name, fig_title, fig_name)
    #
    # # --- for validation dataset
    # num = 50
    # save_file_name = 'results/P2_val.csv'
    # fig_title = 'Validation Dataset'
    # fig_name = 'results/P2_val.png'
    # generate_dateset(num, save_file_name, fig_title, fig_name)
    #
    # # --- for test dataset
    # num = 1000
    # save_file_name = 'results/P2_test.csv'
    # fig_title = 'Test Dataset'
    # fig_name = 'results/P2_test.png'
    # generate_dateset(num, save_file_name, fig_title, fig_name)


    #----------- training, validation, and test
    #--- for Equal Weights
    train_file = 'results/P2_train.csv'
    val_file = 'results/P2_val.csv'
    test_file = 'results/P2_test.csv'
    fig_name = 'results/P2_equal_weights.png'
    equal_cost_svm(train_file, val_file, test_file, fig_name)

    # --- for UnEqual Weights
    train_file = 'results/P2_train.csv'
    val_file = 'results/P2_val.csv'
    test_file = 'results/P2_test.csv'
    fig_name = 'results/P2_UNequal_weights.png'
    unequal_cost_svm(train_file, val_file, test_file, fig_name)

    pass