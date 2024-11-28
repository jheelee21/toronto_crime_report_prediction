from utils import *

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 lr=0.01, prior_mu=0.0, prior_sigma=0.1):
        super(BNN, self).__init__()
        self.model = nn.Sequential(
            bnn.BayesLinear(in_features=input_dim, out_features=32, prior_mu=prior_mu,
                            prior_sigma=prior_sigma),
            nn.ReLU(),
            bnn.BayesLinear(in_features=32, out_features=16, 
                            prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(),
            bnn.BayesLinear(in_features=16, out_features=output_dim, prior_mu=prior_mu,
                            prior_sigma=prior_sigma),
            nn.Sigmoid()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)

    def forward(self, x):
        return self.model(x)


class BatchNormBNN(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 lr=0.01, prior_mu=0.0, prior_sigma=0.1):
        super(BatchNormBNN, self).__init__()
        self.model = nn.Sequential(
            bnn.BayesLinear(in_features=input_dim, out_features=32, 
                            prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.BatchNorm1d(32),
            bnn.BayesLinear(in_features=32, out_features=16, 
                            prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.BatchNorm1d(16),
            bnn.BayesLinear(in_features=16, out_features=output_dim, 
                            prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.Sigmoid()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)

    def forward(self, x):
        return self.model(x)


def batch_train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size):
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)

    losses = []
    val_losses = []
    val_accuracies = []

    # for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):
        print(f"----- Epoch {epoch + 1} -----")
        avg_loss = 0
        for x, y in train_dl:
            # forward pass
            outputs = model(x)
            bce_loss = model.criterion(outputs, y)
            kl_loss = model.kl_loss(model)
            total_loss = bce_loss + 0.1 * kl_loss
            avg_loss += total_loss.item()

            # backward pass + optimisation
            model.optimizer.zero_grad()
            total_loss.backward()
            model.optimizer.step()
        
            model.train()
        avg_loss /= len(train_dl)
        losses.append(avg_loss)

        val_loss, val_acc = validation(model, X_val, y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    plot_loss(range(num_epochs), losses, val_losses)
    plot_acc(range(num_epochs), val_accuracies)


def train(model, X_train, y_train, X_val, y_val, num_epochs):
    losses = []
    val_losses = []
    val_accuracies = []

    # for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):
        print(f"----- Epoch {epoch + 1} -----")
        # forward pass
        outputs = model(X_train)
        bce_loss = model.criterion(outputs, y_train)
        kl_loss = model.kl_loss(model)
        total_loss = bce_loss + 0.1 * kl_loss
        losses.append(total_loss.item())

        # backward pass + optimisation
        model.optimizer.zero_grad()
        total_loss.backward()
        model.optimizer.step()
    
        model.train()
    
        val_loss, val_acc = validation(model, X_val, y_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    plot_loss(range(num_epochs), losses, val_losses)
    plot_acc(range(num_epochs), val_accuracies)


def validation(model, x, y):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        pred = (outputs >= 0.5).float()
        loss = model.criterion(pred, y)
        accuracy = accuracy_score(y, pred)
        print(f"Validation Loss: {loss.item():.4f}")
        print(f"Validation Accuracy: {accuracy:.2%}")
    
    return loss.item(), accuracy


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_tensor()

    lr = 0.01
    batch_size = 256
    num_epochs = 10
    prior_sigma = 0.1
    prior_mu = 0.

    input_dim = X_train.shape[1]
    output_dim = 1

    model = BNN(input_dim=input_dim, output_dim=output_dim, 
                prior_sigma=prior_sigma, prior_mu=prior_mu, lr=lr)
    # model = BatchNormBNN(input_dim=input_dim, output_dim=output_dim)

    train(model, X_train, y_train, X_val, y_val, num_epochs)
    # batch_train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size)

    