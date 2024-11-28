from src.data import load_data

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FILE_PATH = '../data/toronto_crime_data.csv'
SEED = 311
np.random.seed(SEED)

X_train, y_train, X_val, y_val, X_test, y_test = load_data(FILE_PATH)

# TODO: use a loop to go through different hyperparameters w/ validation set
random_forest = RandomForestClassifier(
    n_estimators=20,
    max_depth=8,
    min_samples_split=5,
    random_state=SEED,
)

random_forest.fit(X_train, y_train)

# evaluate on validation set
val_predictions = random_forest.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# evaluate on test set
test_predictions = random_forest.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}")
