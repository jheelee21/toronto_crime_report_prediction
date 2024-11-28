from utils import *
from models import *

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_tensor()

    lr = 0.01
    batch_size = 256
    num_epochs = 10
    prior_sigma = 0.2
    prior_mu = 0.

    input_dim = X_train.shape[1]
    output_dim = 1

    model = BNN(input_dim=input_dim, output_dim=output_dim, 
                prior_sigma=prior_sigma, prior_mu=prior_mu, lr=lr)
    # model = BatchNormBNN(input_dim=input_dim, output_dim=output_dim)
    # model = SoftmaxBNN(input_dim=input_dim, output_dim=output_dim)

    train(model, X_train, y_train, X_val, y_val, num_epochs)
    # batch_train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size)


if __name__ == '__main__':
    main()

    