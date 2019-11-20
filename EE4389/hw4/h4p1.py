import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt


def generate_data(size, filename):
    dim = 20
    features = list()
    labels = np.zeros(size)
    for i in range(dim):
        features.append(np.random.uniform(low=0,high=1,size=size))
    features = np.asarray(features).reshape(-1,dim)
    y_value = np.sum(features[:,0:10], axis=1)-5
    y_neg_idx = np.where(y_value<0)[0]
    y_pos_idx = np.where(y_value>=0)[0]
    labels[y_neg_idx] = -1
    labels[y_pos_idx] = 1
    labels = labels.reshape(-1,1)
    data = np.hstack((features, labels))
    np.savez(filename,data)
    print(filename,'has been saved succesfully!')


def knn_model(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y):
    maxk = 20
    val_acc = [None]*maxk
    for k in range(maxk):
        neigh = KNeighborsClassifier(n_neighbors=k+1)
        neigh.fit(trndata_X, trndata_Y)
        val_acc[k] = neigh.score(valdata_X, valdata_Y)
    max_acc = max(val_acc)
    max_idx = val_acc.index(max_acc)
    opt_k = max_idx + 1
    # using opt k and training model again
    neigh = KNeighborsClassifier(n_neighbors=opt_k)
    neigh.fit(trndata_X, trndata_Y)
    test_acc = neigh.score(tstdata_X, tstdata_Y)
    print('====================== Result for KNN:')
    print('Opt K=', str(opt_k))
    print('Accuracy on validation data=', str(max_acc))
    print('Accuracy on test data=', str(test_acc))
    print('=======================================')
    print(' ')
    knn_result = [opt_k, test_acc]
    return knn_result





def svm_model(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y):
    c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    val_acc = [None]*len(c)
    for i in range(len(c)):
        cur_c = c[i]
        clf = SVC(C=cur_c, kernel='linear', class_weight='balanced')
        clf.fit(trndata_X, trndata_Y)
        val_acc[i] = clf.score(valdata_X, valdata_Y)
    max_acc = max(val_acc)
    max_idx = val_acc.index(max_acc)
    opt_c = c[max_idx]
    # train model again with the opt C
    clf = SVC(C=cur_c, kernel='linear', class_weight='balanced')
    clf.fit(trndata_X, trndata_Y)
    test_acc = clf.score(tstdata_X, tstdata_Y)
    print('====================== Result for SVM:')
    print('Opt C=', str(opt_c))
    print('Accuracy on validation data=', str(max_acc))
    print('Accuracy on test data=', str(test_acc))
    print('=======================================')
    print(' ')
    svm_result = [opt_c, test_acc]
    return svm_result






def compare_knn_svm():
    # training data
    trnfile = 'dataset/hw4/p1_trndata.npz'
    trnsize = 50

    if os.path.exists(trnfile):
        print('Delete old file!')
        os.system('rm {}'.format(trnfile))
        time.sleep(1)
    else:
        print('Generating:', trnfile)

    generate_data(trnsize,trnfile)
    trndata = np.load(trnfile)['arr_0']
    trndata_X = trndata[:,0:-1]
    trndata_Y = trndata[:,-1]

    # validation data
    valfile = 'dataset/hw4/p1_valdata.npz'
    valsize = 50

    if os.path.exists(valfile):
        print('Delete old file!')
        os.system('rm {}'.format(valfile))
        time.sleep(1)
    else:
        print('Generating:', valfile)

    generate_data(valsize, valfile)
    valdata = np.load(valfile)['arr_0']
    valdata_X = valdata[:,0:-1]
    valdata_Y = valdata[:,-1]

    # test data
    tstfile = 'dataset/hw4/p1_tstdata.npz'
    tstsize = 1000
    if os.path.exists(tstfile):
        print('Delete old file!')
        os.system('rm {}'.format(tstfile))
        time.sleep(1)
    else:
        print('Generating:', tstfile)
    generate_data(tstsize, tstfile)
    tstdata = np.load(tstfile)['arr_0']
    tstdata_X = tstdata[:,0:-1]
    tstdata_Y = tstdata[:,-1]

    # pre-processing data
    scaler = StandardScaler()
    scaler.fit(trndata_X)
    trndata_X = scaler.transform(trndata_X)
    valdata_X = scaler.transform(valdata_X)
    tstdata_X = scaler.transform(tstdata_X)

    # start to train model
    knn_result = list()
    svm_result = list()
    knn_result = knn_model(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y)
    svm_result = svm_model(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y)
    return knn_result, svm_result

if __name__ == '__main__':
    N_trial = 1000
    final_K = list()
    final_Acc_knn = list()
    final_C = list()
    final_Acc_svm = list()
    for i in range(N_trial):
        print('>>>>>>> {}/{} Trials'.format(str(i+1), str(N_trial)))
        knn_result, svm_result = compare_knn_svm()
        final_K.append(knn_result[0])
        final_Acc_knn.append(knn_result[1])
        final_C.append(svm_result[0])
        final_Acc_svm.append(svm_result[1])
    ############### Draw Bloxplot #################
    plt.boxplot(final_K,showfliers=False)
    names = ['OPT K']
    plt.ylabel('Times')
    plt.xlabel('Iterations')
    plt.xticks([1], names)
    plt.savefig('dataset/hw4/p1_k.png')
    plt.close()

    plt.boxplot(final_C, showfliers=False)
    names = ['OPT C']
    plt.ylabel('Value')
    plt.xlabel('Iterations')
    plt.xticks([1], names)
    plt.savefig('dataset/hw4/p1_c.png')
    plt.close()

    final_Acc_knn = np.asarray(final_Acc_knn).reshape((-1,1))
    final_Acc_svm = np.asarray(final_Acc_svm).reshape((-1,1))
    final_Acc = np.hstack((final_Acc_knn, final_Acc_svm))
    plt.boxplot(final_Acc, showfliers=False)
    names = ['KNN','SVM']
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.xticks([1,2], names)
    plt.savefig('dataset/hw4/p1_acc.png')
    plt.close()


