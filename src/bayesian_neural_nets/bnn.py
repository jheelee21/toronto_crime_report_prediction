import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import *
from bayesian_neural_nets.utils_outdated import *


EPOCH = 500
X_train, y_train, X_val, y_val, X_test, y_test = np_to_tensor(list(load_data()))


def train_model(model, loss, num_epochs, lr=0.01):
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    losses = []
    model.train()
    for e in tqdm(range(num_epochs)):
        for x, y in zip(X_train, y_train):
            optimizer.zero_grad()
            loss_value = loss(model(x), y)
            loss_value.backward()
            losses.append(loss_value.detach().item())
            optimizer.step()
    return losses


def eval_model(model, X, Y):
    model.eval()
    errors = []
    for x, y in zip(X, Y):
        y_hat, _ = model(x)
        errors.append(((torch.squeeze(y_hat) - torch.squeeze(y))**2).detach().numpy())
    
    rmse = np.sqrt(np.mean(np.concatenate(errors, axis=None)))

    acc = calculate_accuracy(y_hat, y)
    print('- Accuracy: %f %%' % acc)
    return round(rmse, 3)


class BaselineTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BaselineTorch, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.act = torch.relu
        self.double()
    
    def forward(self, inputs):
        h = self.hidden_layer(inputs)
        h = self.act(h)
        output = self.out_layer(h)
        return output, None



class LinearBnnTorch():
    def __init__(self, input_dim, output_dim, mu=0, sigma=0.1, hidden_dim=100, learning_rate=0.01, kl=0.1):
        self.model = nn.Sequential(
            bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, 
                            in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, 
                            in_features=hidden_dim, out_features=output_dim),
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = kl

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.double() 


    def train(self, x, y, epoch):
        for _ in tqdm(range(epoch)):
            pre = self.model(x)
            ce = self.ce_loss(pre, y)
            kl = self.kl_loss(self.model)
            cost = ce + self.kl_weight*kl
            
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()        


    def test(self, x, y):
        pre = self.model(x)
        ce = self.ce_loss(pre, y)
        kl = self.kl_loss(self.model)
        _, predicted = torch.max(pre.data, 1)
        acc = calculate_accuracy(predicted, y)
        print('- Accuracy: %f %%' % acc)
        print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))


class BnnTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=None):
        super(BnnTorch, self).__init__()
        n = input_dim * hidden_dim
        self.mu = nn.Parameter(torch.zeros((n), dtype=torch.float32))
        self.rho  = nn.Parameter(torch.log(torch.expm1(torch.ones((n), dtype=torch.float32))))
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.act = activation
        self.hidden_dim = hidden_dim
        self.prior = torch.distributions.Normal(loc=torch.zeros((n), dtype=torch.float32),
                                                scale=torch.ones((n), dtype=torch.float32))
        self.kl_func = torch.distributions.kl.kl_divergence
        self.batch_norm = torch.nn.BatchNorm1d(input_dim)


    def forward(self, inputs):
        inputs = self.batch_norm(inputs)
        q = torch.distributions.Normal(loc=self.mu, 
                                       scale=torch.log(1.+torch.exp(self.rho)))
        
        kl = torch.sum(self.kl_func(q, self.prior))
        w = q.rsample() 
        w = w.reshape((-1, self.hidden_dim))
        h = inputs @ w
        if self.act is not None:
            h = self.act(h)
        output = self.out_layer(h)
        return output, kl
    

class FullBnnTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=None):
        super(FullBnnTorch, self).__init__()
        n = input_dim * hidden_dim + hidden_dim + hidden_dim + 1
        self.mu = nn.Parameter(torch.zeros((n), dtype=torch.float32))
        self.rho  = nn.Parameter(torch.log(torch.expm1(torch.ones((n), dtype=torch.float32))))
        
        self.act = activation
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.prior = torch.distributions.Normal(loc=torch.zeros((n), dtype=torch.float32),
                                                scale=torch.ones((n), dtype=torch.float32))
  
        self.kl_func = torch.distributions.kl.kl_divergence
        self.batch_norm = torch.nn.BatchNorm1d(input_dim)

        self.double()

        
    def forward(self, inputs):
        inputs = self.batch_norm(inputs)
        q = torch.distributions.Normal(loc=self.mu, 
                                       scale=torch.log(1.+torch.exp(self.rho)))
        
        kl = torch.sum(self.kl_func(q, self.prior))
        all_w = q.rsample() 
        
        W_hidden = all_w[0:self.input_dim * self.hidden_dim].reshape((self.input_dim, self.hidden_dim))
        cur = self.input_dim * self.hidden_dim
        b_hidden = all_w[cur:cur + self.hidden_dim]
        cur = cur + self.hidden_dim
        W_output = all_w[cur:cur + self.hidden_dim].reshape((self.hidden_dim, 1))
        b_output = all_w[-2:-1]
        h = inputs @ W_hidden + b_hidden
        if self.act is not None:
            h = self.act(h)
        output = h @ W_output + b_output
        return output, kl


def compute_predictions(model, X, Y, iterations=20):
    model.eval()
    predicted = np.zeros((y_test.shape[0], iterations))
    
    for j, (x, y) in tqdm(enumerate([X, Y])):
        for i in range(iterations):
            y_hat, _ = model(x)
            predicted[j,i] = y_hat.detach().item()
    return predicted


def main():
    baseline_torch_model = BaselineTorch(len(FEATURES), 32)
    rmse = eval_model(baseline_torch_model, X_val, y_val)
    print(f"untrained RMSE: {rmse:.3f}")
    losses = train_model(baseline_torch_model, simple_mse_loss, num_epochs=1)
    plt.plot(losses)

    # bnn_torch = BnnTorch(11, 32, torch.nn.functional.relu)
    # losses = train_model(bnn_torch, mse_kl_loss, 200)
    # plt.plot(losses)

    # linear_bnn = LinearBnnTorch(len(FEATURES), TARGET_DIM)
    # linear_bnn.train(X_train, y_train, EPOCH)
    # linear_bnn.test(X_val, y_val)


if __name__ == '__main__':
    main()