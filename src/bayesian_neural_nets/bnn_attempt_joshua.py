from src.data import load_df_data

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torchbnn as bnn

# WE DO NOT USE load_data (WHICH INCLUDES 0...N-1 ENCODING)
X_train, y_train, X_val, y_val, X_test, y_test = load_df_data()

# preprocessing for numerical and categorical features
num_features = ["REPORT_YEAR", "REPORT_MONTH", "REPORT_DAY", "REPORT_DOW", "REPORT_DOY", "REPORT_HOUR", "LONGITUDE", "LATITUDE", "AVG_AGE", "POPULATION", "INCOME", "EMPLOYMENT_RATE"]
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

categorical_encoder = preprocessor.named_transformers_['cat']  # Extract the OneHotEncoder

# define our BNN
class CrimeRiskBNN(nn.Module):
    def __init__(self, input_dim, output_dim, prior_mu=0.0, prior_sigma=0.1):
        super(CrimeRiskBNN, self).__init__()
        self.bnn_linear1 = bnn.BayesLinear(in_features=input_dim, out_features=32, prior_mu=prior_mu,
                                           prior_sigma=prior_sigma)
        self.bnn_linear2 = bnn.BayesLinear(in_features=32, out_features=16, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bnn_linear3 = bnn.BayesLinear(in_features=16, out_features=output_dim, prior_mu=prior_mu,
                                           prior_sigma=prior_sigma)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bnn_linear1(x))
        x = self.relu(self.bnn_linear2(x))
        x = self.sigmoid(self.bnn_linear3(x))
        return x

# convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

# instantiate our BNN
model = CrimeRiskBNN(input_dim=X_train.shape[1], output_dim=1)
# we use cross-entropy loss because we are doing binary classification
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# training loop
num_epochs = 10
for epoch in range(num_epochs):
    # forward pass
    outputs = model(X_train_tensor)
    bce_loss = criterion(outputs, y_train_tensor)
    kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)(model)
    total_loss = bce_loss + 0.1 * kl_loss

    # backward pass + optimisation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # validation performance
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_predictions = (val_outputs >= 0.5).float()
        val_loss = criterion(val_predictions, y_val_tensor)
        accuracy = accuracy_score(y_val, val_predictions)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        print(f"Test Accuracy: {accuracy:.2%}")

    model.train()
