from torch.nn.functional import mse_loss
import torch.nn


def completion_network_loss(input, output):
    return mse_loss(input, output)


def l1_loss(input, output):
    loss = torch.nn.L1Loss()
    return loss(input,output)