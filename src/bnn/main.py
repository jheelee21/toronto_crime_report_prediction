from bnn.utils import *
from bnn.models import *


def optimization(model, x, y, scoring='accuracy'):
    print("Optimizing hyperparameters...")

    search_space = {
        "lr": np.arange(0.01, 0.11, 0.02),
        "prior_sigma": np.arange(0.05, 0.3, 0.05),
        "prior_mu": np.arange(-0.03, 0.03, 0.01),
        "kl_weight": np.arange(0.01, 0.5, 0.05)
    }
    results = grid_search(model, search_space, x, y, scoring)
    
    return results


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = load_tensor()

    num_epochs = 30
    # batch_size = 32

    params = {
        "input_dim": X_train.shape[1],
        "output_dim": 1,
        "prior_sigma": 0.15,
        "prior_mu": 0.,
        "lr": 0.05,
        "kl_weight": 0.01
    }

    # we used BNN model for this experiment, but can also use BatchNormBNN and SGDBNN
    model = BNN(**params)    

    # results = optimization(model, X_train, y_train)
    # for k, v in results.loc[0, 'params'].items():
    #     model.__setattr__(k, v)

    train(model, X_train, y_train, X_val, y_val, num_epochs)
    # batch_train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size)

    eval_performance(model, X_test, y_test, dataset='Test', print_val=True)


if __name__ == '__main__':
    main()