from src.data import load_data

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 311
np.random.seed(SEED)

X_train, y_train, X_val, y_val, X_test, y_test = load_data()

# for grid search
Max_depth = [2, 10, 100]
Min_samples_split = [2, 10, 100]

# best performer init
best_model = None
best_params = {}
best_accuracy = 0

# 2nd-best performer init
runnerup_model = None
runnerup_params = {}

# do the grid search, find best performer
for max_depth in Max_depth:
    for min_samples_split in Min_samples_split:
        print(f"Testing: max_depth={max_depth}, min_samples_split={min_samples_split}")

        decision_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=SEED
        )
        decision_tree.fit(X_train, y_train)

        # evaluate on the validation set
        val_predictions = decision_tree.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)

        # update if a better model is found
        if val_accuracy > best_accuracy:
            runnerup_model = best_model
            runnerup_params = best_params
            best_model = decision_tree

            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
            }
            best_accuracy = val_accuracy

# evaluate best model on test set
best_test_predictions = best_model.predict(X_test)
best_test_accuracy = accuracy_score(y_test, best_test_predictions)
print(f"Best model test Accuracy: {best_test_accuracy:.2f} with max_depth={best_params['max_depth']}, "
      f"min_samples_split={best_params['min_samples_split']}")

# evaluate runner-up model on test set
runnerup_test_predictions = runnerup_model.predict(X_test)
runnerup_test_predictions = accuracy_score(y_test, runnerup_test_predictions)
print(f"Runner-up model test Accuracy: {runnerup_test_predictions:.2f} with max_depth={runnerup_params['max_depth']}, "
      f"min_samples_split={runnerup_params['min_samples_split']}")
