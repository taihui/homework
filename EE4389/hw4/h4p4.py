import numpy as np
import math
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def generate_data(size, filename):
    dim = 6
    features = list()
    y = list()
    for i in range(dim):
        features.append(np.random.uniform(low=-1.5,high=1.5,size=size))
    features = np.asarray(features).reshape(-1,dim)
    for i in range(size):
        cur_y = 10* math.sin(math.pi * features[i,0] * features[i,1]) + 20*pow((features[i,2]-0.5),2) + 0* features[i,3] + 5*features[i,4] + 0*features[i,5]
        y.append(cur_y)
    y = np.asarray(y).reshape((-1,1))
    data = np.hstack((features,y))
    np.savez(filename,data)
    print(filename,'has been saved succesfully!')


def svm_model(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y):
    epsilon = [0, 2, 4, 6, 8]
    gamma = [pow(2,-5), pow(2,-4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2), pow(2, 3), pow(2, 4), pow(2, 5)]
    C = np.max(trndata_Y) - np.min(trndata_Y)
    MSE = dict()
    for i in range(len(epsilon)):
        cur_eps = epsilon[i]
        for j in range(len(gamma)):
            cur_g = gamma[j]
            clf = SVR(gamma=cur_g, C=C, epsilon=cur_eps)
            clf.fit(trndata_X, trndata_Y)
            valdata_Y_pred = clf.predict(valdata_X)
            mse = mean_squared_error(valdata_Y, valdata_Y_pred)
            MSE[(cur_eps, cur_g)] = mse
    ########### Print Information ###########
    for key in MSE.keys():
        print('#####################')
        print(key)
        print(MSE[key])
        print('#####################')
        print('')
    print('')
    print('')
    print('')
    ########### After training, get the OPT params ###########
    MSE_Sorted = sorted(MSE.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    opt_param = MSE_Sorted[0][0]
    opt_eps = opt_param[0]
    opt_g = opt_param[1]
    clf = SVR(gamma=opt_g, C=C, epsilon=opt_eps)
    clf.fit(trndata_X, trndata_Y)
    test_Y_pred = clf.predict(tstdata_X)
    mse = mean_squared_error(tstdata_Y, test_Y_pred)
    print(mse)


def svm_model_opt(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y):
    epsilon = [0]
    gamma = [pow(2,-4)]
    C = np.max(trndata_Y) - np.min(trndata_Y)
    for i in range(len(epsilon)):
        cur_eps = epsilon[i]
        for j in range(len(gamma)):
            cur_g = gamma[j]
            clf = SVR(gamma=cur_g, C=C, epsilon=cur_eps)
            clf.fit(trndata_X, trndata_Y)
            trndata_Y_pred = clf.predict(trndata_X)
            trn_mse = mean_squared_error(trndata_Y, trndata_Y_pred)
            print('Training MSE=', str(trn_mse))

            valdata_Y_pred = clf.predict(valdata_X)
            val_mse = mean_squared_error(valdata_Y, valdata_Y_pred)
            print('Val MSE=', str(val_mse))

            tstdata_Y_pred = clf.predict(tstdata_X)
            tst_mse = mean_squared_error(tstdata_Y, tstdata_Y_pred)
            print('Tst MSE=', str(tst_mse))
    ########### Print Information ###########





if __name__ == '__main__':
    # training data
    trnfile = 'dataset/hw4/p4_trndata.npz'
    trnsize = 100
    if os.path.exists(trnfile):
        print('Loading:', trnfile)
    else:
        print('Generating:', trnfile)
        generate_data(trnsize, trnfile)
    trndata = np.load(trnfile)['arr_0']
    trndata_X = trndata[:, 0:-1]
    trndata_Y = trndata[:, -1]

    # validation data
    valfile = 'dataset/hw4/p4_valdata.npz'
    valsize = 100
    if os.path.exists(valfile):
        print('Loading:', valfile)
    else:
        print('Generating:', valfile)
        generate_data(valsize, valfile)
    valdata = np.load(valfile)['arr_0']
    valdata_X = valdata[:, 0:-1]
    valdata_Y = valdata[:, -1]

    # test data
    tstfile = 'dataset/hw4/p4_tstdata.npz'
    tstsize = 800
    if os.path.exists(tstfile):
        print('Loading:', tstfile)
    else:
        print('Generating:', tstfile)
        generate_data(tstsize, tstfile)
    tstdata = np.load(tstfile)['arr_0']
    tstdata_X = tstdata[:, 0:-1]
    tstdata_Y = tstdata[:, -1]

    # pre-processing data
    scaler = StandardScaler()
    scaler.fit(trndata_X)
    trndata_X = scaler.transform(trndata_X)
    valdata_X = scaler.transform(valdata_X)
    tstdata_X = scaler.transform(tstdata_X)



    svm_model(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y)
    svm_model_opt(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y)