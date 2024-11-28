from data import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
import matplotlib.pyplot as plt


def load_tensor():
    # WE DO NOT USE load_data (WHICH INCLUDES 0...N-1 ENCODING)
    X_train, y_train, X_val, y_val, X_test, y_test = load_df_data()

    # preprocessing for numerical and categorical features
    num_features = ["REPORT_YEAR", "REPORT_MONTH", "REPORT_DAY", 
                    "REPORT_DOW", "REPORT_DOY", "REPORT_HOUR", 
                    "LONGITUDE", "LATITUDE", "AVG_AGE", "POPULATION", 
                    "INCOME", "EMPLOYMENT_RATE"]
    cat_features = ["PREMISES_TYPE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),  # standardize numerical features
            ("cat", OneHotEncoder(), cat_features),  # one-hot encode categorical features
        ]
    )

    # preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    for x in [X_train, X_val, X_test]:
        normalize(x, norm='l1')

    # plt.hist(X_train)
    # plt.savefig("bnn_train.png")

    # plot_features(X_train)

    categorical_encoder = preprocessor.named_transformers_['cat']  # Extract the OneHotEncoder

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor


def plot_features(x):
    nrows, ncols = 3, 4
    figure, axis = plt.subplots(nrows, ncols, figsize=(20, 15))
    for i in range(nrows):
        for j in range(ncols):
            k = i * ncols + j

            if k >= x.shape[1]:
                break

            feature = FEATURES[k]
            d = x[:, k]
            axis[i, j].hist(d, bins='auto')
            axis[i, j].set_title(feature)

    plt.tight_layout()
    plt.savefig("bnn_data.png")


def plot_loss(iteration, losses, val_losses):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(iteration, losses, label="Training Loss", color="orange")
    plt.ylabel("Training Loss")

    plt.subplot(212)
    plt.plot(iteration, val_losses, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")

    plt.savefig("bnn_loss.png")


def plot_acc(iteration, accuracy):
    plt.figure(2)
    plt.plot(iteration, accuracy, label="Validation Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.savefig("bnn_accuracy.png")


if __name__ == '__main__':
    load_tensor()