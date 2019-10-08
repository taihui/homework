import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from tabulate import _table_formats, tabulate
import operator

############# Get Dataset #############
def get_dataset():
    dataset_file = 'dataset/hw2/lifeExpectancy.xlsx'
    df = pd.read_excel(dataset_file)
    selected_df = df.iloc[:,2:6]
    print(selected_df.head())
    ### show histograme
    selected_df.hist()
    plt.savefig('dataset/hw2/p3_feature_hist.png')
    plt.show()
    selected_df_np = selected_df.to_numpy()

    #### draw the box-plot for each feature to see the range of their values
    plt.boxplot(selected_df_np, showfliers=False)
    names = ['MURDER','HSGRAD','INCOME','ILLITERACY']
    plt.xticks([1,2,3,4], names)
    plt.savefig('dataset/hw2/p3_boxplot_value_feature.png')
    plt.show()
    selected_df = df.iloc[:, 1:6]
    selected_df_np = selected_df.to_numpy()
    print(selected_df.head())
    return selected_df_np


def linear_regression(dataset, scaling_method = 'standard'):
    target_values = dataset[:, 0]
    feature_values = dataset[:, 1:5]
    if scaling_method == 'standard':
        scaler = StandardScaler().fit(feature_values)
        feature_values_scaled = scaler.transform(feature_values)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler().fit(feature_values)
        feature_values_scaled = scaler.transform(feature_values)
    else:
        print('Scaling method is not supported')
        exit()
    model = LinearRegression()
    model.fit(feature_values_scaled, target_values)
    target_values_pred = model.predict(feature_values_scaled)
    mse = mean_squared_error(target_values, target_values_pred)
    params = model.get_params()
    print('===== Result for LR')
    print(model.coef_)
    print(model.intercept_)
    return params, mse


def linear_regression_cv(dataset, scaling_method = 'standard'):
    target_values = dataset[:, 0]
    feature_values = dataset[:, 1:5]
    MSE_Error = list()
    MSE_Error_train = list()
    cv = 5
    for _ in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(feature_values, target_values, test_size = 0.33, random_state = 42)
        if scaling_method == 'standard':
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            print('Scaling method is not supported')
            exit()
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        mse_train = mean_squared_error(y_train, y_train_pred)
        y_test_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_test_pred)
        MSE_Error_train.append(mse_train)
        MSE_Error.append(mse)
    # After Cross-Validation, get the mean of mse
    MSE_Error_train=np.asarray(MSE_Error_train)
    MSE_Error = np.asarray(MSE_Error)
    MES_Mean_train = np.mean(MSE_Error_train)
    MSE_Mean = np.mean(MSE_Error)
    final_mse = np.hstack((MSE_Error_train.reshape(-1,1), MSE_Error.reshape(-1,1)))
    ap_df = pd.DataFrame({'Training MSE': final_mse[:, 0],
                          'Test MSE': final_mse[:, 1],
                          })
    ap_file = 'dataset/hw2/p3_b.csv'
    ap_df.to_csv(ap_file, index=False)
    print(np.mean(final_mse[:, 0]))
    print(np.mean(final_mse[:, 1]))
    return MSE_Mean, MES_Mean_train

def expectancy_murder(dataset, scaling_method = 'standard'):
    target_values = dataset[:, 0].reshape((-1,1))
    murder_values = dataset[:,1].reshape((-1,1))
    if scaling_method == 'standard':
        scaler = StandardScaler().fit(murder_values)
        murder_values_scaled = scaler.transform(murder_values)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler().fit(murder_values)
        murder_values_scaled = scaler.transform(murder_values)
    else:
        print('Scaling method is not supported')
        exit()
    model = LinearRegression()
    model.fit(murder_values_scaled, target_values)
    target_values_pred = model.predict(murder_values_scaled)
    mse = mean_squared_error(target_values, target_values_pred)
    params = model.get_params()
    # plot figure
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(murder_values, target_values), key=sort_axis)
    X_real_sorted, Y_real_sorted = zip(*sorted_zip)

    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(murder_values, target_values_pred), key=sort_axis)
    X_pred_sorted, Y_pred_sorted = zip(*sorted_zip)

    plt.scatter(murder_values, target_values, marker="+", c='k')
    #plt.plot(X_real_sorted, Y_real_sorted, ls='-.', label='Ground Truth')
    plt.plot(X_pred_sorted, Y_pred_sorted, label='Estimated Function')
    plt.xlabel('Murder Rate')
    plt.ylabel('Life Expectancy')
    plt.legend()
    plt.savefig('dataset/hw2/p3_expectancy_murder_rate.png')
    print(model.coef_)
    print(model.intercept_)
    return params, mse


def check_correration():
    dataset_file = 'dataset/hw2/lifeExpectancy.xlsx'
    df = pd.read_excel(dataset_file)
    selected_df = df.iloc[:, 1:6]
    correlation_matrix = selected_df.corr().round(2)
    print(correlation_matrix)
    sns.heatmap(data=correlation_matrix, annot=True)

    f, ax = plt.subplots(figsize=(5, 5))
    ax = sns.heatmap(correlation_matrix, ax=ax, vmin=0, vmax=1, annot=True, fmt='0.1g')

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=45, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.show()







    plt.savefig('dataset/hw2/p3_correration_heatmap.png', bbox_inches='tight')
    plt.show()
    plt.close()

    print(selected_df.head())
    corr_matrix = selected_df.corr()
    corr_sorted = corr_matrix['LIFE EXPECT.'].sort_values(ascending=False)
    # draw figures
    from pandas.plotting import scatter_matrix
    attributes = ["LIFE EXPECT.", 'MURDER', 'HSGRAD', 'INCOME', 'ILLITERACY']
    scatter_matrix(selected_df[attributes], figsize=(12, 8))
    plt.savefig('dataset/hw2/p3_correration.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    ### a
    #dataset = get_dataset()
    #linear_regression(dataset, scaling_method='standard')

    ### b
    #dataset = get_dataset()
    #mse = linear_regression_cv(dataset, scaling_method='standard')

    ### c
    #dataset = get_dataset()
    #expectancy_murder(dataset, scaling_method='standard')

    get_dataset()

    pass