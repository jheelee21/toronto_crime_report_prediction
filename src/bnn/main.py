from utils import *
from models import *

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = load_tensor()

    lr = 0.01
    batch_size = 256
    num_epochs = 10
    prior_sigma = 0.2
    prior_mu = 0.

    input_dim = X_train.shape[1]
    output_dim = 1

    # model = BNN(input_dim=input_dim, output_dim=output_dim, 
    #             prior_sigma=prior_sigma, prior_mu=prior_mu, lr=lr)
    # model = BatchNormBNN(input_dim=input_dim, output_dim=output_dim)
    model = BNN(input_dim=input_dim, output_dim=output_dim,
                       prior_sigma=prior_sigma, prior_mu=prior_mu, lr=lr)

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

    infer(model, new_data_all, preprocessor)

if __name__ == '__main__':
    main()

    