from data import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


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

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, preprocessor


def infer(model, features, preprocessor, num_samples=100):
    X_df = pd.DataFrame([features])
    X = preprocessor.transform(X_df)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()
    predictions = []

    # perform multiple forward passes
    with torch.no_grad():
        for _ in range(num_samples):
            prediction = model(X_tensor).item()
            predictions.append(prediction)

    # compute mean and standard deviation of predictions
    predictions_tensor = torch.tensor(predictions)
    mean_prediction = predictions_tensor.mean().item()
    std_prediction = predictions_tensor.std().item()

    print(f"Prediction Mean: {mean_prediction:.4f}")
    print(f"Prediction Std Dev: {std_prediction:.4f}")
    print(f"Predicted Class: {'High-risk' if mean_prediction >= 0.5 else 'Low-risk'}")

    return mean_prediction, std_prediction


def train(model, X_train, y_train, X_val, y_val, num_epochs):
    train_losses = []
    train_accuracies = []
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
        train_losses.append(total_loss.item())

        # backward pass + optimisation
        model.optimizer.zero_grad()
        total_loss.backward()
        model.optimizer.step()
    
        model.train()
    
        _, train_acc = eval_performance(model, X_train, y_train, "Train")
        train_accuracies.append(train_acc)
        val_loss, val_acc = eval_performance(model, X_val, y_val, "Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    plot_loss(range(num_epochs), train_losses, val_losses)
    plot_train_val_acc(range(num_epochs), train_accuracies, val_accuracies)


def batch_train(model, X_train, y_train, X_val, y_val, num_epochs, batch_size):
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):
        print(f"----- Epoch {epoch + 1} -----")
        for x, y in train_dl:
            # forward pass
            outputs = model(x)
            bce_loss = model.criterion(outputs, y)
            kl_loss = model.kl_loss(model)
            total_loss = bce_loss + 0.1 * kl_loss

            # backward pass + optimisation
            model.optimizer.zero_grad()
            total_loss.backward()
            model.optimizer.step()
        
            model.train()

        train_loss, train_acc = eval_performance(model, X_train, y_train, "Train")
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = eval_performance(model, X_val, y_val, "Validation")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    plot_loss(range(num_epochs), train_losses, val_losses)
    plot_train_val_acc(range(num_epochs), train_accuracies, val_accuracies)


def eval_performance(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        pred = (outputs >= 0.5).float()
        loss = model.criterion(pred, y)
        accuracy = accuracy_score(y, pred)
        print(f"{criterion} Loss: {loss.item():.4f} | {criterion} Accuracy: {accuracy:.2%}")
    
    return loss.item(), accuracy


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

    plt.savefig("bnn/bnn_train_val_loss.png")


def plot_acc(iteration, accuracy, criterion):
    plt.figure(2)
    plt.plot(iteration, accuracy)
    plt.xlabel("Iteration")
    plt.ylabel(f"{criterion} Accuracy")
    plt.savefig("bnn/bnn_accuracy.png")

def plot_train_val_acc(iteration, train_acc, val_acc):
    plt.figure(2)
    plt.subplot(211)
    plt.plot(iteration, train_acc, color='orange')
    plt.ylabel("Train Accuracy")

    plt.subplot(212)
    plt.plot(iteration, val_acc)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")

    plt.savefig("bnn/bnn_train_val_accuracy.png")


if __name__ == '__main__':
    load_tensor()