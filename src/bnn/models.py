from utils import *

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
    

class SoftmaxBNN(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 lr=0.01, prior_mu=0.0, prior_sigma=0.1):
        super(SoftmaxBNN, self).__init__()
        self.model = nn.Sequential(
            bnn.BayesLinear(in_features=input_dim, out_features=32, 
                            prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(),
            bnn.BayesLinear(in_features=32, out_features=16, 
                            prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(),
            bnn.BayesLinear(in_features=16, out_features=output_dim, 
                            prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.Softmax(dim=0)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)

    def forward(self, x):
        return self.model(x)