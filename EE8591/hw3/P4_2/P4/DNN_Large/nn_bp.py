import torch

from util import *
from tqdm import tqdm
import pandas as pd
from classifier_model import *
import copy
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use("seaborn")
#plt.style.use("ggplot")
#plt.style.use("bmh")
plt.style.use("seaborn-darkgrid")
#plt.style.use("seaborn-deep")
#plt.style.use("tableau-colorblind10")


def deepreg_train(learning_rate,max_epoch,train_batch_size,val_batch_size,test_batch_size,gpu,print_step,opm_tag,train_f5,train_f8,val_f5,val_f8,test_f5,test_f8):

    dir_name = 'deepnn/training_results/'
    dir_model_name = 'deepnn/training_models'
    dir_name_test = 'deepnn/test_results/'

    make_dir([dir_name, dir_model_name, dir_name_test])

    ################## Using GPU when it is available ##################
    if gpu=="MSI":
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    ################## Get Dataset ##################
    train_X_5 = np.load(train_f5)['arr_0']
    train_Y_5 = 0 * np.ones(len(train_X_5)).reshape(-1, 1)
    train_X_8 = np.load(train_f8)['arr_0']
    train_Y_8 = np.ones(len(train_X_8)).reshape(-1, 1)
    train_X = np.vstack((train_X_5, train_X_8)).reshape((-1, 1, 28, 28))
    train_Y = np.vstack((train_Y_5, train_Y_8))

    # val
    val_X_5 = np.load(val_f5)['arr_0']
    val_Y_5 = 0 * np.ones(len(val_X_5)).reshape(-1, 1)
    val_X_8 = np.load(val_f8)['arr_0']
    val_Y_8 = np.ones(len(val_X_8)).reshape(-1, 1)
    val_X = np.vstack((val_X_5, val_X_8)).reshape((-1, 1, 28, 28))
    val_Y = np.vstack((val_Y_5, val_Y_8))

    # test
    test_X_5 = np.load(test_f5)['arr_0']
    test_Y_5 = 0 * np.ones(len(test_X_5)).reshape(-1, 1)
    test_X_8 = np.load(test_f8)['arr_0']
    test_Y_8 = np.ones(len(test_X_8)).reshape(-1, 1)
    test_X = np.vstack((test_X_5, test_X_8)).reshape((-1, 1, 28, 28))
    test_Y = np.vstack((test_Y_5, test_Y_8))

    train_num = len(train_Y)
    val_num = len(val_Y)
    test_num = len(test_Y)


    train_loader = prepare_data(train_X, train_Y, train_batch_size, train_num)
    val_loader = prepare_data(val_X, val_Y, val_batch_size, val_num)
    test_loader = prepare_data(test_X, test_Y, test_batch_size, test_num)

    ################## Define NN Models ##################
    DLRegNet = MNIST_Classifier()
    DLRegNet.to(device)

    # define loss function & Opt
    criterion = nn.CrossEntropyLoss()

    if opm_tag == "Adam":
        optimizer = torch.optim.Adam(DLRegNet.parameters(), lr= learning_rate)
    elif opm_tag == 'SGD':
        optimizer = torch.optim.SGD(DLRegNet.parameters(), lr=learning_rate)
    else:
        assert False, "Opt Error!"

    ################## Start to Train NN model ##################
    total_loss = []
    total_epoch = []

    train_ACC = []
    val_Acc = []


    best_ACC = -1
    best_train_Acc = -1
    best_epoch = 0


    for epoch in range(max_epoch):
        print('')
        print('')
        print('###################### Start to Train NN model ##########################')
        DLRegNet.train()
        epoch_loss = []
        progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)
        for step, (X_data, Y_labels, idx) in enumerate(train_loader):
            ################## Get Training & Traget Dataset ##################
            X_data = X_data.to(device).float()
            Y_labels = Y_labels.to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            Y_pred = DLRegNet(X_data)
            loss = criterion(Y_pred, Y_labels.squeeze())  # MSE loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.cpu().numpy())
            progress.set_postfix({'loss': loss.data.cpu().numpy()})
            progress.update()
        progress.close()
        total_loss.append(np.mean(epoch_loss))

        # print out the information each "print_step" steps and val our model
        if epoch%print_step == 0:
            cur_train_acc = deep_reg_val(DLRegNet, train_loader, device)
            cur_val_acc = deep_reg_val(DLRegNet, val_loader, device)
            train_ACC.append(cur_train_acc)
            val_Acc.append(cur_val_acc)
            if best_train_Acc <= cur_train_acc:
                best_train_Acc = cur_train_acc
            if best_ACC <= cur_val_acc:
                best_ACC = cur_val_acc
                best_epoch = epoch
                ################ check and always save the best model we have
                model_file_name = os.path.join(dir_model_name, 'net_{}.pt'.format(epoch))
                save_model(DLRegNet.eval(), model_file_name)
            figure_name = os.path.join(dir_name, '1_train_val_mse_{}.png'.format(epoch))
            display_MSE(train_ACC, val_Acc, print_step, figure_name)
            figure_name = os.path.join(dir_name, '0_train_loss_{}.png'.format(epoch))
            display_train_loss(total_loss, figure_name)


    #################### After we finish our training, let test how our model work on test data
    best_model_file = os.path.join(dir_model_name,'net_{}.pt'.format(best_epoch))
    best_Net = MNIST_Classifier()
    try:
        best_Net.load_state_dict(torch.load(best_model_file, map_location='cuda:{}'.format(gpu)))
        print('Loading Pretrained models 1!')
    except:
        best_Net = nn.DataParallel(best_Net)
        best_Net.load_state_dict(torch.load(best_model_file, map_location='cuda:{}'.format(gpu)))
        print('Loading Pretrained models 2!')

    best_Net.to(device)
    best_Net.eval()

    final_train_acc = deep_reg_val(best_Net, train_loader, device)
    final_val_acc = deep_reg_val(best_Net, val_loader, device)
    final_test_acc = deep_reg_val(best_Net, test_loader, device)



    #----------save best epoch for reference
    ####### write information into txt file #########
    txt_name = os.path.join(dir_name_test, '00_best_epoch.txt')
    f = open(txt_name, 'w+')
    f.write('best_epoch={}\n\n'.format(best_epoch))
    f.write('final_train_acc={}\n\n'.format(final_train_acc))
    f.write('final_val_acc={}\n\n'.format(final_val_acc))
    f.write('final_test_acc={}\n\n'.format(final_test_acc))
    f.close()
    print("Finish!")


def deep_reg_val(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (X_data, Y_label, idx) in enumerate(data_loader):
            X_data = X_data.to(device).float()
            Y_label = Y_label.to(device).long()
            Y_pred = model(X_data)
            _, predicted = torch.max(Y_pred.data, 1)
            total += Y_label.size(0)
            correct += (predicted == Y_label).sum().item()
    acc = correct / total
    return acc


if __name__ == '__main__':
    pass





















