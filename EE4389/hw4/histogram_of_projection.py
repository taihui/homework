# This script draws the projection histogram
import matplotlib.pyplot as plt
import numpy as np
from numpy import ones

def draw_hop(trn_decision_value, trn_labels, figurename):
    ######### processing trn_decision_value #########
    #trn_decision_value = trn_decision_value
    trn_decision_value = trn_decision_value.reshape(-1, 1)
    trn_labels = trn_labels.reshape(-1,1)



    trn_decision_value_cls1 = trn_decision_value[np.where(trn_labels==float(-1))[0], 0]
    trn_decision_value_cls1 = trn_decision_value_cls1.reshape(-1,1)

    trn_decision_value_cls2 = trn_decision_value[np.where(trn_labels==float(1))[0], 0]
    trn_decision_value_cls2 = trn_decision_value_cls2.reshape(-1,1)

    # get numbers
    trn_cls1_num = trn_decision_value_cls1.shape[0]
    trn_cls2_num = trn_decision_value_cls2.shape[0]

    # get hist
    trn_cls1_hist, trn_cls1_edge = np.histogram(trn_decision_value_cls1, bins='auto')

    trn_cls2_hist, trn_cls2_edge = np.histogram(trn_decision_value_cls2, bins='auto')

    # normalize hist
    trn_cls1_hist = trn_cls1_hist/trn_cls1_num
    trn_cls2_hist = trn_cls2_hist/trn_cls2_num

    # get mid_height

    mid_height = (max([np.amax(trn_cls1_hist), np.amax(trn_cls2_hist)]) + min([np.amin(trn_cls1_hist), np.amin(trn_cls2_hist)])) / 2
    trn_cls1h = mid_height * ones((trn_cls1_num,1))
    trn_cls2h = mid_height * ones((trn_cls2_num,1))

    ######### Draw figures #########
    trn_cls1_edge = calculate_center(trn_cls1_edge)
    trn_cls2_edge = calculate_center(trn_cls2_edge)


    # draw figure split
    plt.scatter(trn_decision_value_cls1, trn_cls1h, marker='*', color='b')
    plt.scatter(trn_decision_value_cls2, trn_cls2h, marker='+', color='m')

    plt.plot(trn_cls1_edge, trn_cls1_hist,label='cls1', color='b', lw=2.0)
    plt.plot(trn_cls2_edge, trn_cls2_hist,label='cls2', color='m', lw=2.0)

    plt.plot([0, 0], [0, max([np.amax(trn_cls1_hist), np.amax(trn_cls2_hist)])], 'k')
    plt.plot([-1, -1], [0, max([np.amax(trn_cls1_hist), np.amax(trn_cls2_hist)])], 'k-.')
    plt.plot([+1, +1], [0, max([np.amax(trn_cls1_hist), np.amax(trn_cls2_hist)])], 'k-.')

    plt.legend(loc='best')
    plt.savefig(figurename)
    plt.close()



def calculate_center(input_array):
    first_array = input_array[1:len(input_array)]
    second_array = input_array[0:-1]
    sum_array = first_array + second_array
    sum_array = sum_array/2
    return sum_array


if __name__ == '__main__':

    pass
