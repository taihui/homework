import torch.nn as nn
import torch
import torch.nn.functional as F


################ Define Classifer Using MLP ################
class MyClassifier(nn.Module):
    def __init__(self):
        input_feature = 784
        output_feature = 2
        neuron_num = 1024
        super(MyClassifier, self).__init__()
        self.clf_net = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=neuron_num, bias=True),
            nn.BatchNorm1d(neuron_num),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.5),
            nn.Linear(in_features=neuron_num, out_features=neuron_num, bias=True),
            nn.BatchNorm1d(neuron_num),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.5),
            nn.Linear(in_features=neuron_num, out_features=neuron_num, bias=True),
            nn.BatchNorm1d(neuron_num),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            nn.Linear(in_features=neuron_num, out_features=output_feature, bias=True),
        )

    def forward(self, x):
        output = self.clf_net(x)
        return output