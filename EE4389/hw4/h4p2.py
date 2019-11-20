from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from histogram_of_projection import *
import os
from util import *
from sklearn.utils import shuffle
from smallestenclosingcircle import *
from sklearn.decomposition import PCA
import math
import miniball


def generate_data(size, filename):
    noise = np.random.normal(0, 0.025, size)
    # negative samples
    x1 = np.random.uniform(low=0.2,high=0.6,size=size)
    x1 = x1.reshape((-1,1))
    x2 = list()
    for i in range(size):
        cur_feature = pow((x1[i]-0.4)*3,2) + 0.225 + noise[i]
        x2.append(cur_feature)
    x2 = np.asarray(x2)
    x2 = x2.reshape((-1,1))
    neg_y = -1*np.ones(size).reshape((-1,1))
    neg_features = np.hstack((x1,x2,neg_y))

    # positive samples
    x1 = np.random.uniform(low=0.4, high=0.8, size=size)
    x1 = x1.reshape((-1,1))
    x2 = list()
    for i in range(size):
        cur_feature = pow((x1[i] - 0.4) * 3, 2) + 0.225 + noise[i]
        x2.append(cur_feature)
    x2 = np.asarray(x2)
    x2 = x2.reshape((-1,1))
    pos_y = np.ones(size).reshape((-1, 1))
    pos_features = np.hstack((x1, x2, pos_y))

    data = np.vstack((neg_features, pos_features))
    np.savez(filename, data)
    print(filename, 'has been saved succesfully!')



################## For Error Bound ###############################
################## For Error Bound ###############################
################## For Error Bound ###############################
def svm_model_hyper_errbound(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y,figurename):
    c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    gamma = [pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2), pow(2, 3), pow(2, 4),
             pow(2, 5), pow(2, 6), pow(2, 7), pow(2, 8)]
    Acc = dict()
    for c_item in c:
        for g_item in gamma:
            clf = SVC(gamma=g_item, C=c_item, kernel='rbf')
            clf.fit(trndata_X, trndata_Y)
            sup_num = len(clf.support_ )
            total_num = len(trndata_Y)
            bound_err = sup_num/total_num
            #val_acc = clf.score(valdata_X, valdata_Y)
            Acc[((c_item, g_item))] = bound_err
    ########## Choose the Opt param ##########
    Acc_Sorted = sorted(Acc.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    opt_param = Acc_Sorted[0][0]
    opt_c = opt_param[0]
    opt_g = opt_param[1]
    print('================ OPT Params for Hyper ErrBound ================')
    print('gamma=', str(opt_g))
    print('C=', str(opt_c))
    clf = SVC(gamma=opt_g, C=opt_c, kernel='rbf')
    clf.fit(trndata_X, trndata_Y)
    test_acc = clf.score(tstdata_X, tstdata_Y)
    print("Test Acc =", str(test_acc))
    trn_decision_value = clf.decision_function(trndata_X)
    tst_decision_value = clf.decision_function(tstdata_X)
    draw_hop(trn_decision_value, trndata_Y, figurename[0])
    draw_hop(tst_decision_value, tstdata_Y, figurename[1])
    print('Figure has been saved!')


def svm_model_mnist_errbound(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y,figurename):
    c = [0.01, 0.1, 1, 10, 100, 1000]
    gamma = [pow(2,-8), pow(2,-6), pow(2,-4), pow(2, -2), 1, pow(2, 2), pow(2, 4)]
    Acc = dict()
    for c_item in c:
        for g_item in gamma:
            clf = SVC(gamma=g_item, C=c_item, kernel='rbf')
            clf.fit(trndata_X, trndata_Y)
            sup_num = len(clf.support_ )
            total_num = len(trndata_Y)
            bound_err = sup_num / total_num
            # val_acc = clf.score(valdata_X, valdata_Y)
            Acc[((c_item, g_item))] = bound_err
    ########## Choose the Opt param ##########
    Acc_Sorted = sorted(Acc.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    opt_param = Acc_Sorted[0][0]
    opt_c = opt_param[0]
    opt_g = opt_param[1]
    print('================ OPT Params for MNIST ErrBound================')
    print('gamma=', str(opt_g))
    print('C=', str(opt_c))
    clf = SVC(gamma=opt_g, C=opt_c, kernel='rbf')
    clf.fit(trndata_X, trndata_Y)
    test_acc = clf.score(tstdata_X, tstdata_Y)
    print("Test Acc =", str(test_acc))
    trn_decision_value = clf.decision_function(trndata_X)
    tst_decision_value = clf.decision_function(tstdata_X)
    draw_hop(trn_decision_value, trndata_Y, figurename[0])
    draw_hop(tst_decision_value, tstdata_Y, figurename[1])
    print('Figure has been saved!')


################## For VC  ###############################
################## For VC  ###############################
################## For VC  ###############################
def svm_model_hyper_vc(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y,figurename):
    c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    gamma = [pow(2,-5), pow(2,-4), pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 1), pow(2, 2), pow(2, 3), pow(2, 4), pow(2, 5),pow(2, 6),pow(2, 7),pow(2, 8)]
    Acc = dict()
    for c_item in c:
        for g_item in gamma:
            clf = SVC(gamma=g_item, C=c_item, kernel='rbf')
            clf.fit(trndata_X, trndata_Y)
            remp = 1 - clf.score(trndata_X, trndata_Y)
            support_vector = clf.support_vectors_
            support_cof = clf.dual_coef_.reshape((-1,1))
            support_idx = clf.support_
            support_y = trndata_Y[support_idx]
            update_vector = clf.support_vectors_.copy()
            for i in range(len(support_y)):
                cof_val = support_cof[i][0]
                y_val = support_y[i]
                update_vector[i,:] = y_val * cof_val * support_vector[i,:]
            w = np.sum(update_vector, axis=0)
            vc_err = vc_bound(trndata_X, w, remp)
            Acc[((c_item, g_item))] = vc_err
    ########## Choose the Opt param ##########
    Acc_Sorted = sorted(Acc.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    opt_param = Acc_Sorted[0][0]
    opt_c = opt_param[0]
    opt_g = opt_param[1]
    print('================ OPT Params for Hyper VCBound================')
    print('gamma=', str(opt_g))
    print('C=', str(opt_c))
    clf = SVC(gamma=opt_g, C=opt_c, kernel='rbf')
    clf.fit(trndata_X, trndata_Y)
    test_acc = clf.score(tstdata_X, tstdata_Y)
    print("Test Acc =", str(test_acc))
    trn_decision_value = clf.decision_function(trndata_X)
    tst_decision_value = clf.decision_function(tstdata_X)
    draw_hop(trn_decision_value, trndata_Y, figurename[0])
    draw_hop(tst_decision_value, tstdata_Y, figurename[1])
    print('Figure has been saved!')


def svm_model_mnist_vc(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y,figurename):
    c = [0.01, 0.1, 1, 10, 100, 1000]
    gamma = [pow(2,-8), pow(2,-6), pow(2,-4), pow(2, -2), 1, pow(2, 2), pow(2, 4)]
    Acc = dict()
    C, radius = miniball.get_bounding_ball(trndata_X)
    for c_item in c:
        print('c=',str(c_item))
        for g_item in gamma:
            print('g=',str(g_item))
            clf = SVC(gamma=g_item, C=c_item, kernel='rbf')
            clf.fit(trndata_X, trndata_Y)
            remp = 1 - clf.score(trndata_X, trndata_Y)
            w = clf.dual_coef_
            vc_err = vc_bound(trndata_X, w, remp,radius)
            Acc[((c_item, g_item))] = vc_err
    ########## Choose the Opt param ##########
    Acc_Sorted = sorted(Acc.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
    opt_param = Acc_Sorted[0][0]
    opt_c = opt_param[0]
    opt_g = opt_param[1]
    print('================ OPT Params for MNIST VCBound================')
    print('gamma=', str(opt_g))
    print('C=', str(opt_c))
    clf = SVC(gamma=opt_g, C=opt_c, kernel='rbf')
    clf.fit(trndata_X, trndata_Y)
    test_acc = clf.score(tstdata_X, tstdata_Y)
    print("Test Acc =", str(test_acc))
    trn_decision_value = clf.decision_function(trndata_X)
    tst_decision_value = clf.decision_function(tstdata_X)
    draw_hop(trn_decision_value, trndata_Y, figurename[0])
    draw_hop(tst_decision_value, tstdata_Y, figurename[1])
    print('Figure has been saved!')

def vc_bound(trndata_X,w, remp, radius):

    delta = 1/np.linalg.norm(w)
    num = trndata_X.shape[0]
    d = trndata_X.shape[1]
    h = min((radius*radius)/(delta*delta),d) + 1
    eng = min(4/math.sqrt(num),1)
    upper = h*(np.log(2*num/h)+1) - np.log(eng/4)
    conf = math.sqrt(upper/num)
    vc_error = remp + conf
    return vc_error










if __name__ == '__main__':
    ############################### Hyperbolas ###############################
    # training data
    trnfile = 'dataset/hw4/p2_trndata.npz'
    trnsize = 50
    if os.path.exists(trnfile):
        print('Loading:', trnfile)
    else:
        print('Generating:', trnfile)
        generate_data(trnsize, trnfile)
    trndata = np.load(trnfile)['arr_0']
    trndata_X = trndata[:, 0:-1]
    trndata_Y = trndata[:, -1]

    # validation data
    valfile = 'dataset/hw4/p2_valdata.npz'
    valsize = 50
    if os.path.exists(valfile):
        print('Loading:', valfile)
    else:
        print('Generating:', valfile)
        generate_data(valsize, valfile)
    valdata = np.load(valfile)['arr_0']
    valdata_X = valdata[:, 0:-1]
    valdata_Y = valdata[:, -1]

    # test data
    tstfile = 'dataset/hw4/p2_tstdata.npz'
    tstsize = 1000
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



    figurename1 = 'dataset/hw4/p2_hop_hyperbolas_train_errbound.png'
    figurename2 = 'dataset/hw4/p2_hop_hyperbolas_tst_errbound.png'
    figurename = [figurename1, figurename2]
    #svm_model_hyper_errbound(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y, figurename)

    figurename1 = 'dataset/hw4/p2_hop_hyperbolas_train_vcbound.png'
    figurename2 = 'dataset/hw4/p2_hop_hyperbolas_tst_vcbound.png'
    figurename = [figurename1, figurename2]
    #svm_model_hyper_vc(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y, figurename)

    ############################### MNIST ###############################
    training_dataset, test_dataset = get_dataset()
    trn_idx5 = training_dataset.targets == 5
    trn_X5 = training_dataset.data[trn_idx5].numpy().reshape(-1,28*28)
    trn_X5 = trn_X5/255

    trn_idx8 = training_dataset.targets == 8
    trn_X8 = training_dataset.data[trn_idx8].numpy().reshape(-1,28*28)
    trn_X8 = trn_X8 / 255

    tst_idx5 = test_dataset.targets == 5
    tst_X5 = test_dataset.data[tst_idx5].numpy().reshape(-1,28*28)
    tst_X5 = tst_X5 / 255

    tst_idx8 = test_dataset.targets == 8
    tst_X8 = test_dataset.data[tst_idx8].numpy().reshape(-1,28*28)
    tst_X8 = tst_X8 / 255

    # training data
    trndata_X5 = trn_X5[0:500,:]
    trndata_Y5 = -1 * np.ones(500)
    trndata_X8 = trn_X8[0:500, :]
    trndata_Y8 = np.ones(500)
    trndata_X = np.vstack((trndata_X5, trndata_X8))
    trndata_Y = np.hstack((trndata_Y5, trndata_Y8))

    # validation data
    valdata_X5 = trn_X5[500:1000, :]
    valdata_Y5 = -1 * np.ones(500)
    valdata_X8 = trn_X8[500:1000, :]
    valdata_Y8 = np.ones(500)
    valdata_X = np.vstack((valdata_X5, valdata_X8))
    valdata_Y = np.hstack((valdata_Y5, valdata_Y8))

    # test data
    tstdata_X5 = tst_X5
    tstdata_Y5 = -1*np.ones(len(tstdata_X5))
    tstdata_X8 = tst_X8
    tstdata_Y8 = np.ones(len(tstdata_X8))

    tstdata_X = np.vstack((tstdata_X5, tstdata_X8))
    tstdata_Y = np.hstack((tstdata_Y5, tstdata_Y8))


    scaler = StandardScaler()
    scaler.fit(trndata_X)
    trndata_X = scaler.transform(trndata_X)
    valdata_X = scaler.transform(valdata_X)
    tstdata_X = scaler.transform(tstdata_X)

    figurename1 = 'dataset/hw4/p2_hop_mnist_trn_errbound.png'
    figurename2 = 'dataset/hw4/p2_hop_mnist_tst_errbound.png'
    figurename = [figurename1, figurename2]
    svm_model_mnist_errbound(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y, figurename)

    figurename1 = 'dataset/hw4/p2_hop_mnist_trn_vcbound.png'
    figurename2 = 'dataset/hw4/p2_hop_mnist_tst_vcbound.png'
    figurename = [figurename1, figurename2]
    svm_model_mnist_vc(trndata_X, trndata_Y, valdata_X, valdata_Y, tstdata_X, tstdata_Y, figurename)
