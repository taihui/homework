import numpy as np
from numpy import mean
from tabulate import _table_formats, tabulate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import operator



########################## Problem 2 ##########################
def generate_dataset(sample_number, iteration):
    X = np.random.uniform(0,1,sample_number)
    noise = np.random.normal(0, 0.5, sample_number)
    Y = pow(X,2) + 0.1 * X + noise
    Y_denoised = pow(X, 2) + 0.1 * X
    return X, Y, Y_denoised

def draw_dataset(X, Y):
    plt.scatter(X, Y, marker="+", c='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Samples')
    plt.savefig('dataset/hw2/p2_train_samples.png')
    plt.close()


def polynomial_regression_cv(X, Y, degree_dict):
    loss_list = list()
    rows = list()
    for i in range(len(degree_dict)):
        degree = degree_dict[i]
        cv = 5
        cv_loss = list()
        for _ in range(cv):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
            polynomial_features = PolynomialFeatures(degree)
            X_poly = polynomial_features.fit_transform(X_train)
            model = LinearRegression()
            model.fit(X_poly, Y_train)
            X_test_poly = polynomial_features.fit_transform(X_test)
            Y_test_pred = model.predict(X_test_poly)
            temp_loss = mean_squared_error(Y_test, Y_test_pred)
            cv_loss.append(temp_loss)
        current_row = list()
        current_row.append(i+1)
        current_row.extend(cv_loss)
        current_row.append(mean(cv_loss))
        rows.append(current_row)
        loss_list.append(mean(cv_loss))
    min_loss_value = min(loss_list)
    min_loss_idx = loss_list.index(min_loss_value)
    opt_degree = degree_dict[min_loss_idx]
    # generate opt information
    polynomial_features = PolynomialFeatures(opt_degree)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, Y)
    Y_poly_pred = model.predict(X_poly)
    opt_loss = mean_squared_error(Y, Y_poly_pred)
    return X_poly, Y_poly_pred, opt_loss, opt_degree, rows

def draw_cv(X, Y, Y_denoised,Y_poly_pred_cv, rows, opt_degree_cv):
    #### print table
    rows = np.asarray(rows)
    headers = ["Degree", "CV-1", "CV-2", "CV-3", "CV-4", "CV-5", "Mean"]
    format_list = list(_table_formats.keys())
    print(tabulate(rows, headers, tablefmt='fancy_grid'))
    # for f in format_list:
    # print("\nformat: {}\n".format(f))
    #### draw figure
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_denoised), key=sort_axis)
    X_denoised_sorted, Y_denoised_sorted = zip(*sorted_zip)

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_poly_pred_cv), key=sort_axis)
    X_cv_sorted, Y_cv_sorted = zip(*sorted_zip)
    plt.scatter(X, Y, marker="+", c='k')
    plt.plot(X_denoised_sorted, Y_denoised_sorted, ls='-.', label='ground truth')
    plt.plot(X_cv_sorted, Y_cv_sorted, label='5-fold cross-validation: degree=' + str(opt_degree_cv))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset/hw2/p2_cv.png')
    plt.close()



def schwartz_crit(param_num, sample_num):
    p = param_num/sample_num
    r = 1 + p * pow((1-p),-1) * np.log(sample_num)
    return r

def polynomial_regression_schwartz(X,Y, degree_dict):
    loss_list = list()
    rows = list()
    for i in range(len(degree_dict)):
        degree = degree_dict[i]
        polynomial_features = PolynomialFeatures(degree)
        X_poly = polynomial_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, Y)
        Y_poly_pred = model.predict(X_poly)
        mse = mean_squared_error(Y, Y_poly_pred)
        r = schwartz_crit(degree+1, X.shape[0])
        current_loss = r * mse
        loss_list.append(current_loss)
        current_row = list()
        current_row.append(i+1)
        current_row.append(current_loss)
        rows.append(current_row)
    min_loss_value = min(loss_list)
    min_loss_idx = loss_list.index(min_loss_value)
    opt_degree = degree_dict[min_loss_idx]
    # generate opt information
    polynomial_features = PolynomialFeatures(opt_degree)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, Y)
    Y_poly_pred = model.predict(X_poly)
    opt_loss = mean_squared_error(Y, Y_poly_pred)
    return X_poly, Y_poly_pred, opt_loss, opt_degree, rows

def draw_sc(X, Y, Y_denoised, Y_poly_pred_sc, opt_degree_sc, rows):
    # draw figures
    # sort the values of x before line plot
    rows = np.asarray(rows)
    headers = ["Degree", "Loss"]
    format_list = list(_table_formats.keys())
    print(tabulate(rows, headers, tablefmt='fancy_grid'))

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_denoised), key=sort_axis)
    X_denoised_sorted, Y_denoised_sorted = zip(*sorted_zip)

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_poly_pred_sc), key=sort_axis)
    X_sc_sorted, Y_sc_sorted = zip(*sorted_zip)

    plt.scatter(X, Y, marker="+", c='k')
    plt.plot(X_denoised_sorted, Y_denoised_sorted, ls='-.', label='ground truth')
    plt.plot(X_sc_sorted, Y_sc_sorted, label='Schwartz criterion: degree=' + str(opt_degree_sc))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset/hw2/p2_sc.png')
    plt.close()

def draw_cv_sc(X, Y, Y_denoised,Y_poly_pred_cv, opt_degree_cv, Y_poly_pred_sc, opt_degree_sc):
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_denoised), key=sort_axis)
    X_denoised_sorted, Y_denoised_sorted = zip(*sorted_zip)

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_poly_pred_cv), key=sort_axis)
    X_cv_sorted, Y_cv_sorted = zip(*sorted_zip)

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_poly_pred_sc), key=sort_axis)
    X_sc_sorted, Y_sc_sorted = zip(*sorted_zip)


    plt.scatter(X, Y, marker="+", c='k')
    plt.plot(X_denoised_sorted, Y_denoised_sorted, ls='-.', label='ground truth')
    plt.plot(X_cv_sorted, Y_cv_sorted, label='5-fold cross-validation: degree=' + str(opt_degree_cv))
    plt.plot(X_sc_sorted, Y_sc_sorted, label='Schwartz criterion: degree=' + str(opt_degree_sc))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset/hw2/p2_cv_sc.png')
    plt.close()




def polynomial_regression_double_resampling(X, Y, degree_dict):
    test_erlist, validation_erlist, opt_dgs = list(), list(), list()
    loo = X.shape[0]
    for i in range(loo):
        loo_train_X = np.delete(X, i).reshape((-1,1))
        loo_train_Y = np.delete(Y, i).reshape((-1,1))
        loo_test_X = X[i].reshape((-1,1))
        loo_test_Y = Y[i].reshape((-1,1))
        all_cv_loss = list()
        for dg_idx in range(len(degree_dict)):
            degree = degree_dict[dg_idx]
            val_cv = 6
            cv_loss = list()
            for _ in range(val_cv):
                val_train_X, val_test_X, val_train_Y, val_test_Y = train_test_split(loo_train_X, loo_train_Y, test_size = 0.2, random_state = 42)
                polynomial_features = PolynomialFeatures(degree)
                val_poly_X = polynomial_features.fit_transform(val_train_X)
                model = LinearRegression()
                model.fit(val_poly_X, val_train_Y)
                val_test_X_poly = polynomial_features.fit_transform(val_test_X)
                val_poly_pred_Y = model.predict(val_test_X_poly)
                temp_loss = mean_squared_error(val_test_Y, val_poly_pred_Y)
                cv_loss.append(temp_loss)
            all_cv_loss.append(mean(cv_loss))
        # find the opt degree for this loo
        min_loss_value = min(all_cv_loss)
        min_loss_idx = all_cv_loss.index(min_loss_value)
        opt_degree = degree_dict[min_loss_idx]
        opt_dgs.append(opt_degree)
        validation_erlist.append(min_loss_value)
        # use this opt degree to train a model on loo_train and then test on loo_test
        polynomial_features = PolynomialFeatures(opt_degree)
        loo_poly_X = polynomial_features.fit_transform(loo_train_X)
        model = LinearRegression()
        model.fit(loo_poly_X, loo_train_Y)
        loo_test_X_poly = polynomial_features.fit_transform(loo_test_X)
        loo_poly_pred_Y = model.predict(loo_test_X_poly)
        loo_loss = mean_squared_error(loo_test_Y, loo_poly_pred_Y)
        test_erlist.append(loo_loss)
    return test_erlist, validation_erlist, opt_dgs

def draw_double(test_erlist, validation_erlist, opt_dgs):
    ### Save each iteration result to csv file
    final_ap = np.hstack((np.asarray(opt_dgs).reshape((-1, 1)),
                          np.asarray(validation_erlist).reshape((-1, 1)),
                          np.asarray(test_erlist).reshape((-1, 1)),
                          ))
    #### print table
    headers = ["Optimal degree", "6-cv validation error", "LOO test error"]
    format_list = list(_table_formats.keys())
    print(tabulate(final_ap, headers, tablefmt='fancy_grid'))

    ap_df = pd.DataFrame({'Optimal degree': final_ap[:, 0],
                          '6-cv validation error': final_ap[:, 1],
                          'LOO test error': final_ap[:, 2],
                          })
    ap_file = 'dataset/hw2/p2_double.csv'
    ap_df.to_csv(ap_file, index=False)
    print(np.mean(final_ap[:, 0]))
    print(np.mean(final_ap[:, 1]))
    print(np.mean(final_ap[:, 2]))


def compare_everything(iteration, showfigure=False):
    all_loss_cv = list()
    all_dg_cv = list()
    all_loss_sc = list()
    all_dg_sc = list()
    all_loss_double = list()
    all_dg_double = list()
    for i in range(iteration):
        print('========='+str(i+1)+'/'+str(1000))
        X, Y, Y_denoised = generate_dataset(25, -1)
        X = X.reshape((-1, 1))
        Y = Y.reshape((-1, 1))
        Y_denoised = Y_denoised.reshape((-1, 1))
        degree_dict = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 5-cv
        X_poly_cv, Y_poly_pred_cv, opt_loss_cv, opt_degree_cv, rows_cv = polynomial_regression_cv(X, Y, degree_dict)
        if showfigure == True:
            draw_cv(X, Y, Y_denoised, Y_poly_pred_cv, rows_cv, opt_degree_cv)
        # sc
        X_poly_sc, Y_poly_pred_sc, opt_loss_sc, opt_degree_sc, rows_sc = polynomial_regression_schwartz(X,Y, degree_dict)
        if showfigure == True:
            draw_sc(X, Y, Y_denoised, Y_poly_pred_sc, opt_degree_sc, rows_sc)
            draw_cv_sc(X, Y, Y_denoised,Y_poly_pred_cv, opt_degree_cv, Y_poly_pred_sc, opt_degree_sc)
        # double
        test_erlist, validation_erlist, opt_dgs = polynomial_regression_double_resampling(X, Y, degree_dict)
        if showfigure == True:
            draw_double(test_erlist, validation_erlist, opt_dgs)
        #
        all_loss_cv.append(opt_loss_cv)
        all_dg_cv.append(opt_degree_cv)

        all_loss_sc.append(opt_loss_sc)
        all_dg_sc.append(opt_degree_sc)

        all_loss_double.append(np.mean(test_erlist))
        all_dg_double.append(np.mean(opt_dgs))
    #### Draw box-plot to compare
    ## cv-sc
    ### After 1000 iterations, show the box-plot for cv and sc
    final_loss = np.hstack((np.asarray(all_loss_cv).reshape((-1, 1)), np.asarray(all_loss_sc).reshape((-1, 1))))
    plt.boxplot(final_loss, showfliers=False)
    # plt.xlabel('Method')
    plt.ylabel('MSE')
    names = ['CV', 'Schwartz']
    plt.xticks([1, 2], names)
    plt.savefig('dataset/hw2/p2_boxplot_cv_sc_loss.png')
    plt.close()

    final_degree = np.hstack((np.asarray(all_dg_cv).reshape((-1, 1)), np.asarray(all_dg_sc).reshape((-1, 1))))
    plt.boxplot(final_degree, showfliers=False)
    plt.ylabel('Degree')
    names = ['CV', 'Schwartz']
    plt.xticks([1, 2], names)
    plt.savefig('dataset/hw2/p2_boxplot_cv_sc_degree.png')
    plt.close()

    #### double-sc
    ### After 1000 iterations, show the box-plot for double and sc
    final_loss = np.hstack((np.asarray(all_loss_double).reshape((-1, 1)), np.asarray(all_loss_sc).reshape((-1, 1))))
    plt.boxplot(final_loss, showfliers=False)
    # plt.xlabel('Method')
    plt.ylabel('MSE')
    names = ['Double resampling','Schwartz']
    plt.xticks([1, 2], names)
    plt.savefig('dataset/hw2/p2_boxplot_double_sc_loss.png')
    plt.close()

    final_degree = np.hstack((np.asarray(all_dg_double).reshape((-1, 1)), np.asarray(all_dg_sc).reshape((-1, 1))))
    plt.boxplot(final_degree, showfliers=False)
    plt.ylabel('Degree')
    names = ['Double resampling','Schwartz']
    plt.xticks([1, 2], names)
    plt.savefig('dataset/hw2/p2_boxplot_double_sc_degree.png')
    plt.close()








if __name__ == '__main__':
    compare_everything(100, showfigure=False)
    print('Finish!')
    pass
