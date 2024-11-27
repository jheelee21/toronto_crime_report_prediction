import matplotlib.pyplot as plt
import torch

def calculate_accuracy(pred, y):
    total = y.size(0)
    correct = (pred == y).sum()
    accuracy = 100 * float(correct) / total
    return accuracy

def simple_mse_loss(model_outputs, y_true):
    y_hat, _ = model_outputs
    y_hat = torch.squeeze(y_hat)
    y_true = torch.squeeze(y_true)
    return torch.nn.MSELoss()(y_hat, y_true) 


def mse_kl_loss(model_outputs, y_true):
    kl_weight = 1. / len(y_true)
    y_hat, kl = model_outputs
    y_hat = torch.squeeze(y_hat)
    y_true = torch.squeeze(y_true)
    mse = torch.nn.MSELoss()(y_hat, y_true)
    return mse + kl * kl_weight

