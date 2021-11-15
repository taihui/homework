import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


###################################################################
################# Prepare AE Dataset into Batch Size #################
###################################################################
class RegDataset(Dataset):
    def __init__(self, clean_data, input_label, num, transform=None):
        self.transform = transform
        self.clean_data = clean_data
        self.label = input_label
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample_clean = self.clean_data[idx,:]
        sample_label = self.label[idx]
        if self.transform:
            print('No transform is needed!')
            #sample_clean = self.transform(sample_clean)
        sample_clean = torch.from_numpy(sample_clean)
        return (sample_clean, sample_label, idx)

#################  #################
def prepare_data(input_data, input_label, batch_size, num):
    reg_dataset = RegDataset(input_data, input_label, num)
    dataloader = DataLoader(reg_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

###################################################################
##################### Define Loss Function #########################
###################################################################
def L2_Func():
    return torch.nn.MSELoss()

def L1_Func():
    return torch.nn.L1Loss()

def L1_Smooth_Func():
    return torch.nn.SmoothL1Loss()

def Pseudo_Huber_Loss(true_data, pred_data, delta, device):
    t = torch.abs(true_data - pred_data)
    flag = torch.tensor(delta).to(device)
    ret = torch.where(flag==delta, delta **2 *((1+(t/delta)**2)**0.5-1), t)
    mean_loss = torch.mean(ret)
    return mean_loss

def Huber_Loss(true_data, pred_data, delta):
    t = torch.abs(true_data - pred_data)
    ret = torch.where(t <= delta, 0.5 * t ** 2, delta * t - 0.5 * delta ** 2)
    mean_loss = torch.mean(ret)
    return mean_loss


def get_L1_L2(check_matrix):
    cur_l1_norm = torch.norm(check_matrix, p=1)
    cur_l2_norm = torch.norm(check_matrix, p=2)
    measurement = cur_l1_norm / cur_l2_norm
    return measurement
###################################################################
##################### Save Model and Code #########################
###################################################################
def save_model(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)
    print('The trained model has been saved!')

def save_code(code, code_file_name):
    np.savez_compressed(code_file_name, code)
    print('The updated code has been saved!')

def make_dir(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            os.system('rm {}*'.format(dir))

###################################################################
##################### custom weights initialization ###############
###################################################################
def weights_init(m):
    classname = m.__class__.__name__
    # initialize Linear layers
    if classname.find('Linear') != -1:
        #torch.nn.init.normal_(m.weight.data, 0.0, 1e-4)
        #torch.nn.init.kaiming_normal_(m.weight.data)
        print('Initialize Linear layers!')

    if classname.find('embedding') != -1:
        # torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.normal_(m.weight.data, 0.0, 1)
        print('Initialize Z layers!')

    # initialize Conv/Deconv layers
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
        print('Initialize Conv/Deconv layers!')
    # initialize Bathnorm layers
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
        print('Initialize Bathnorm layers!')

###################################################################
########################### plot figures ###########################
###################################################################
################### Handle Loss ###################
def display_train_loss(train_loss,loss_file='training_results/1_loss.png'):
    plt.plot(train_loss,label='Train Loss')
    plt.ylabel('MSE')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(loss_file)
    plt.close()

def display_MSE(train_mse, val_mse, print_step, loss_file='training_results/1_loss.png'):
    plt.plot(train_mse,label='Train MSE')
    plt.plot(val_mse, label='Val MSE')
    plt.xlabel('Epoch X{}'.format(print_step))
    plt.ylabel('MSE')
    plt.title('Train vs. Val')
    plt.legend()
    plt.savefig(loss_file)
    plt.close()



def save_loss(total_loss,loss_file = 'training_results/corrupted_train_loss.npz'):
    print('Final loss = {}'.format(total_loss[-1]))
    total_loss = np.asarray(total_loss)
    np.savez(loss_file, total_loss)
    print('Loss has been saved!')


if __name__ == '__main__':
    pass








