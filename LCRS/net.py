from __future__ import print_function
import torch
from model.Bin_LeNet import Bin_LeNet_train
from model.LeNet import LeNet
import torch.nn as nn

class LCRS(nn.Module):
    def __init__(self, in_channels, out_channels, num_branchs):
        super(LCRS, self).__init__()
        self.num_branch = num_branchs
        self.binary_models = []
        for _ in range(num_branchs):
            self.binary_models.append(Bin_LeNet_train(in_channels, out_channels))
        self.binary_models = nn.ModuleList(self.binary_models)
        self.main_model = LeNet(in_channels,out_channels)
    def forward(self, x):
        B = x.shape[0]
        hs, predictions = [], []
        for i, binary_model in enumerate(self.binary_models):
            h, prediction = binary_model(x[:, i])
            hs.append(h)
            predictions.append(prediction)

        h = torch.cat(hs, dim=1)
        h = self.main_model(h)
        h = self.pool(h)
        prediction = self.classifier(h.view(B, -1))
        predictions.append(prediction)
        return predictions
