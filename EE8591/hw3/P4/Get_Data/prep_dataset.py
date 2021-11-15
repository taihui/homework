import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np


def save_code(code, code_file_name):
    np.savez_compressed(code_file_name, code)
    print('The updated code has been saved!')


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

train_idx_5 = mnist_trainset.train_labels==5
train_label_5 = mnist_trainset.train_labels[train_idx_5].numpy()
train_data_5 = mnist_trainset.train_data[train_idx_5].numpy()

train_idx_8 = mnist_trainset.train_labels==8
train_label_8 = mnist_trainset.train_labels[train_idx_8].numpy()
train_data_8 = mnist_trainset.train_data[train_idx_8].numpy()


mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

test_idx_5 = mnist_testset.test_labels==5
test_label_5 = mnist_testset.test_labels[test_idx_5].numpy()
test_data_5 = mnist_testset.test_data[test_idx_5].numpy()


test_idx_8 = mnist_testset.test_labels==8
test_label_8 = mnist_testset.test_labels[test_idx_8].numpy()
test_data_8 = mnist_testset.test_data[test_idx_8].numpy()



############## save dataset
### for small dataset
small_train_file_5 = 'small_train_5.npz'
small_train_file_8 = 'small_train_8.npz'
small_val_file_5 = 'small_val_5.npz'
small_val_file_8 = 'small_val_8.npz'
test_file_5 = 'test_5.npz'
test_file_8 = 'test_8.npz'
num = 10
#----------train 5 & 8
small_train_X_5 = train_data_5[0:10]/255
save_code(small_train_X_5, small_train_file_5)
small_train_X_8 = train_data_8[0:10]/255
save_code(small_train_X_8, small_train_file_8)

#----------val 5 & 8
small_val_X_5 = train_data_5[10:20]/255
save_code(small_val_X_5, small_val_file_5)
small_val_X_8 = train_data_8[10:20]/255
save_code(small_val_X_8, small_val_file_8)

#----------test 5 & 8
test_X_5 = test_data_5/255
save_code(test_X_5, test_file_5)
test_X_8 = test_data_8/255
save_code(test_X_8, test_file_8)


#----------------for large dataset
large_train_file_5 = 'large_train_5.npz'
large_train_file_8 = 'large_train_8.npz'
large_val_file_5 = 'large_val_5.npz'
large_val_file_8 = 'large_val_8.npz'

#----------train 5 & 8
large_train_X_5 = train_data_5[0:400]/255
save_code(large_train_X_5, large_train_file_5)
large_train_X_8 = train_data_8[0:400]/255
save_code(large_train_X_8, large_train_file_8)

#----------val 5 & 8
large_val_X_5 = train_data_5[400:800]/255
save_code(large_val_X_5, large_val_file_5)
large_val_X_8 = train_data_8[400:800]/255
save_code(large_val_X_8, large_val_file_8)