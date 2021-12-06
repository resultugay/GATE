import torch
import torch.nn as nn

class A2VLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(A2VLoss, self).__init__()

    def forward(self, input_, target_, smooth=1):
        return torch.exp(input_)

class PairWiseLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PairWiseLoss, self).__init__()

    def forward(self,res, target_):
        return (1-(res[0]) + (1+res[1]))