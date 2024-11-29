from .utils import *
from .models import *

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = load_tensor()

    num_epochs = 50
    batch_size = 32
    params = {
        "input_dim": X_train.shape[1],
        "output_dim": 1,
        "prior_sigma": 0.1,
        "prior_mu": 0.,
        "lr": 0.05,
        "kl_weight": 0.01
    }
    # params = {
    #     "input_dim": X_train.shape[1],
    #     "output_dim": 1,
    #     "prior_sigma": 0.15,
    #     "prior_mu": 0.,
    #     "lr": 0.05,
    #     "kl_weight": 0.01
    # }

    model = BNN(**params)    

    # results = optimization(model, X_train, y_train)
    # for k, v in results.loc[0, 'params'].items():
    #     model.__setattr__(k, v)

    train(model, X_train, y_train, X_val, y_val, num_epochs)
    # batch_train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size)

    # sample input to infer
    new_data_all = {
        'REPORT_YEAR': 2023,
        'REPORT_MONTH': 1,
        'REPORT_DAY': 3,
        'REPORT_DOW': 1,
        'REPORT_DOY': 269,
        'REPORT_HOUR': 13,
        'PREMISES_TYPE': 'Apartment',
        'LONGITUDE': -79.6066,
        'LATITUDE': 43.7441,
        'AVG_AGE': 40,
        'POPULATION': 33300,
        'INCOME': 20000,
        'EMPLOYMENT_RATE': 80
    }

    # infer(model, new_data_all, preprocessor)

if __name__ == '__main__':
    main()

    