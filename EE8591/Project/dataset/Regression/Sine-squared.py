# this script will be used to generated the empirical dataset
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_dataset_A(sample_num, dataset_name):
    X = np.random.uniform(low=0.0, high=1.0, size=sample_num)
    noise = np.random.normal(loc=0.0, scale=0.1, size=sample_num)
    Y_clean = np.sin(2*np.pi*X)**2
    Y = Y_clean + noise

    df = pd.DataFrame()
    df['X'] = X
    df['Y_clean'] = Y_clean
    df['Y'] = Y
    df.to_csv(dataset_name, index=False)

    ### let's visualize the dataset we just generated
    plt.figure()
    plt.scatter(X, Y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sine-squared dataset ({} samples)'.format(sample_num))
    figure_name = 'Sine-squared_{}.png'.format(sample_num)
    plt.savefig(figure_name, dpi=300)
    plt.close()

    print('Congrats! The dataset has been saved successfully!')

    X = X.reshape((-1,1))
    Y = Y.reshape((-1, 1))
    Y_clean = Y_clean.reshape((-1, 1))

    return X, Y, Y_clean

def generate_dataset_B(sample_num, dataset_name):
    X = np.random.uniform(low=0.0, high=1.0, size=sample_num)
    Y = np.random.normal(loc=0.0, scale= 1, size=sample_num)
    Y_clean = np.zeros(sample_num)



    df = pd.DataFrame()
    df['X'] = X
    df['Y_clean'] = Y_clean
    df['Y'] = Y
    df.to_csv(dataset_name, index=False)

    ### let's visualize the dataset we just generated
    plt.figure()
    plt.scatter(X, Y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('The generated dataset B ({} samples)'.format(sample_num))
    figure_name = 'Dataset_B_{}.png'.format(sample_num)
    plt.savefig(figure_name, dpi=300)
    plt.close()

    print('Congrats! The dataset has been saved successfully!')

    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    Y_clean = Y_clean.reshape((-1, 1))

    return X, Y, Y_clean


if __name__ == '__main__':

    ############# Dataset for A
    # sample_num = 25
    # dataset_name = 'Dataset_A_{}.csv'.format(sample_num)
    # generate_dataset_A(sample_num, dataset_name)

    sample_num = 50
    dataset_name = 'Sine-squared_{}.csv'.format(sample_num)
    generate_dataset_A(sample_num, dataset_name)

    sample_num = 100
    dataset_name = 'Sine-squared_{}.csv'.format(sample_num)
    generate_dataset_A(sample_num, dataset_name)
    
    
    sample_num = 500
    dataset_name = 'Sine-squared_{}.csv'.format(sample_num)
    generate_dataset_A(sample_num, dataset_name)

    # ############# Dataset for B
    # sample_num = 25
    # dataset_name = 'Dataset_B_{}.csv'.format(sample_num)
    # generate_dataset_B(sample_num, dataset_name)
    #
    # sample_num = 50
    # dataset_name = 'Dataset_B_{}.csv'.format(sample_num)
    # generate_dataset_B(sample_num, dataset_name)
    #
    # sample_num = 100
    # dataset_name = 'Dataset_B_{}.csv'.format(sample_num)
    # generate_dataset_B(sample_num, dataset_name)
