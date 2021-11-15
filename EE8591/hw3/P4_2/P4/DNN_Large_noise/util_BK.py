import torch
import torch.utils.data as Data
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import math
import skimage as sk
from sklearn.decomposition import PCA


###################################################################
################# Prepare Dataset into Batch Size #################
###################################################################
class MNIST(Dataset):
    def __init__(self, clean_data, corrupted_data, class_target, transform=None):
        self.transform = transform
        self.clean_data = clean_data
        self.corrupted_data = corrupted_data
        self.target = class_target


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        sample_label = self.target[idx]
        sample_clean = self.clean_data[idx]
        sample_corrputed = self.corrupted_data[idx]
        if self.transform:
            sample_clean = self.transform(sample_clean)
            sample_corrputed = self.transform(sample_corrputed)
        #print(torch.min(sample_clean))
        #print(torch.max(sample_clean))
        return (sample_clean, sample_corrputed, idx, sample_label)


#################  #################
#transforms.Normalize(mean=(0.1307,),std=(0.3081,))
def prepare_mnist(batch_size, clean_file,corrupted_file,target_file):
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
    ])

    clean_data, corrupted_data, class_target = create_clean_corrupted_dataset(clean_file,corrupted_file,target_file)
    mnist_dataset = MNIST(clean_data, corrupted_data, class_target, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataloader, mnist_dataset



###################################################################
############ Process Dataset and Corrupted Dataset ################
###################################################################
# here we download the test dataset instead of training dataset
def create_clean_corrupted_dataset(clean_file,corrupted_file,target_file):
    # this function will load the dataset we want
    clean_data = np.load(clean_file)['arr_0']
    corrupted_data = np.load(corrupted_file)['arr_0']
    class_target = np.load(target_file)['arr_0']
    return clean_data, corrupted_data, class_target









###################################################################
########## define corruption function here ########################
###################################################################


###################################################################
##################### Prepare Latent Code #########################
###################################################################
def prepare_code(train_loader, code_dim):
    # initialize representation space
    Z = np.empty((len(train_loader.dataset), code_dim))
    Z = np.random.randn(len(train_loader.dataset), code_dim)
    return Z


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
        torch.nn.init.kaiming_normal_(m.weight.data)
        print('Initialize Linear layers!')
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
def display_loss(total_loss, loss_file='training_results/1_loss.png'):
    plt.plot(total_loss,label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_file)
    plt.close()

################### Handle Loss ###################
def display_acc_top1(acc_clean, acc_cor, acc_rec, total_epoch, loss_file='training_results/1_acc.png'):
    plt.plot(acc_clean,label='Clean')
    plt.plot(acc_cor, label='Cor')
    plt.plot(acc_rec, label='Rec')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Top-1 Accuracy')
    custom_y_ticks = np.arange(0, 1.1, 0.1)
    plt.yticks(custom_y_ticks)
    index_ls = total_epoch
    scale_ls = range(len(acc_clean))
    _ = plt.xticks(scale_ls, index_ls)
    plt.grid()
    plt.legend()
    plt.savefig(loss_file)
    plt.close()

def display_acc_top5(acc_clean, acc_cor, acc_rec, total_epoch, loss_file='training_results/1_acc.png'):
    plt.plot(acc_clean,label='Clean')
    plt.plot(acc_cor, label='Cor')
    plt.plot(acc_rec, label='Rec')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Top-5 Accuracy')
    custom_y_ticks = np.arange(0, 1.1, 0.1)
    plt.yticks(custom_y_ticks)
    index_ls = total_epoch
    scale_ls = range(len(acc_clean))
    _ = plt.xticks(scale_ls, index_ls)
    plt.grid()
    plt.legend()
    plt.savefig(loss_file)
    plt.close()

def save_loss(total_loss,loss_file = 'training_results/corrupted_train_loss.npz'):
    print('Final loss = {}'.format(total_loss[-1]))
    total_loss = np.asarray(total_loss)
    np.savez(loss_file, total_loss)
    print('Loss has been saved!')

################### Draw Figures ###################
def display_image_example(image_data):
    plt.imshow(image_data)
    plt.show()
    plt.close()

def draw_figures(all_images, figure_name):
    #all_images = torch.cat([true_images, fake_images], dim=0)
    plt.figure(figsize=(28, 28))
    plt.axis("off")
    plt.title("Real Images (1st row), Corrupted Images (2nd row), Reconstructed Images (3rd row)")
    plt.imshow(np.transpose(vutils.make_grid(all_images, nrow=10, padding=2, normalize=True),(1,2,0)))
    plt.savefig(figure_name)
    plt.close()
    #plt.show()

def display_acc_top(acc_clean, acc_cor, acc_rec, total_epoch, loss_file='training_results/1_acc.png'):
    plt.plot(acc_clean,label='Clean')
    plt.plot(acc_cor, label='Cor')
    plt.plot(acc_rec, label='Rec')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    custom_y_ticks = np.arange(0, 1.1, 0.1)
    plt.yticks(custom_y_ticks)
    index_ls = total_epoch
    scale_ls = range(len(acc_clean))
    _ = plt.xticks(scale_ls, index_ls)
    plt.grid()
    plt.legend()
    plt.savefig(loss_file)
    plt.close()




if __name__ == '__main__':
    pass








