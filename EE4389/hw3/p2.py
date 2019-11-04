import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pandas as pd
import random
import operator
from sklearn.preprocessing import StandardScaler,MinMaxScaler


############### Define MLP Model ###############
class MLP_Regression(nn.Module):
    def __init__(self, hid_nodes):
        super(MLP_Regression,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features = 1, out_features=hid_nodes, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features = hid_nodes, out_features = 1, bias = True),
            nn.Sigmoid(),
            nn.Linear(in_features=1, out_features=1, bias= True)
        )

    def forward(self,X):
        return self.main(X)


# custom weights initialization called on netG and netD
def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        #nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

############### Generate Dataset ###############
def generate_data(sample_size):
    X = np.random.uniform(0,1,sample_size)
    noise = np.random.normal(0, 0.5, sample_size)
    Y = X ** 2 + 0.1 * X + noise
    Y_denoised = X ** 2 + 0.1 * X
    final_data = np.hstack((np.asarray(X).reshape((-1,1)), np.asarray(Y).reshape((-1,1)), np.asarray(Y_denoised).reshape((-1,1))))
    df = pd.DataFrame({'X': final_data[:, 0],
                          'Y': final_data[:, 1],
                          'Y_denoised': final_data[:, 2],
                         })
    data_file = 'dataset/hw3/data.csv'
    df.to_csv(data_file, index=False)
    plt.scatter(X, Y, marker="+")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig('dataset/hw3/dataset.png')
    plt.close()





############### Create Dir ###############
def make_dir():
    dirs = ['dataset/hw3/', 'dataset/']
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

def train(hid_nodes):
    ############# Set Up #############
    make_dir()
    sample_size = 25
    LR = 0.0005
    #hid_nodes = 8
    batch_size = 25
    EPOCH = 5000
    ############# Get Dataset #############
    data_file = 'dataset/hw3/data.csv'
    if os.path.exists(data_file):
        pass
    else:
        generate_data(sample_size)
    df = pd.read_csv(data_file)
    rawX = df['X'].to_numpy().reshape((-1,1))
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    scaler.fit(rawX)
    X = scaler.transform(rawX)
    data_size = len(X)
    Y = df['Y'].to_numpy()
    Y_denoised = df['Y_denoised'].to_numpy()
    ############# Create MLP Model #############
    mlp_reg_model = MLP_Regression(hid_nodes)
    #mlp_reg_model.apply(weights_init)
    print(mlp_reg_model)

    ############# Define Loss Function and Optimizer #############
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_reg_model.parameters(), lr=LR )

    ############# Start to Train #############
    total_loss = list()
    for epoch in range(EPOCH):
        batch_num = int(data_size/batch_size)
        epoch_loss = list()
        for i in range(batch_num):
            idxs = range(0, data_size)
            selected_idxs = random.sample(idxs, batch_size)
            selected_X = X[selected_idxs]
            X_tensor = torch.from_numpy(selected_X).view((-1,1))
            selected_Y = Y[selected_idxs]
            Y_tensor = torch.from_numpy(selected_Y).view((-1,1))
            optimizer.zero_grad()
            pred_Y = mlp_reg_model(X_tensor.float())
            loss = loss_func(pred_Y.float(), Y_tensor.float())
            loss.backward()
            optimizer.step()
            print('>>>>>>Epoch:{}/{}, Batch:{}/{}, loss={}'.format(str(epoch+1), str(EPOCH),str(i+1),str(batch_num),str(loss.item())))
            epoch_loss.append(loss.item())
        total_loss.append(np.mean(epoch_loss))
    # draw loss figure
    plt.plot(range(len(total_loss)),total_loss)
    plt.xlabel('Number of epochs')
    plt.ylabel('MSE(loss)')
    plt.savefig('dataset/hw3/p2_loss_{}.png'.format(hid_nodes))
    plt.close()
    # draw figure for the model prediction
    X_tensor = torch.from_numpy(X).view((-1,1))
    with torch.no_grad():  # we don't need gradients in the testing phase
        pred_Y = mlp_reg_model(X_tensor.float())
        pred_Y = pred_Y.detach().numpy()

    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, Y_denoised), key=sort_axis)
    X_denoised_sorted, Y_denoised_sorted = zip(*sorted_zip)

    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, pred_Y), key=sort_axis)
    X_denoised_sorted, pred_Y_sorted = zip(*sorted_zip)

    plt.scatter(X, Y, marker='+', edgecolors='k')
    plt.plot(X_denoised_sorted, Y_denoised_sorted,color='k',ls='--', label='Ground Truth')
    plt.plot(X_denoised_sorted, pred_Y_sorted, color='r', label='MLP Prediction')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig('dataset/hw3/p2_pred_fig_{}.png'.format(str(hid_nodes)))
    plt.close()
    print('Finish!')
    print(" ")
    print(" ")


if __name__ == '__main__':
    hid_nodes_list = range(1,100)
    for hid_nodes in hid_nodes_list:
        train(hid_nodes)








