from src.data import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test = load_data()


def grid_search():
    # for grid search
    Max_depth = [2, 10, 100]
    Min_samples_split = [2, 10, 100]
    Criterions = ["gini", "entropy", "log_loss"]

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
            for criterion in Criterions:
                print(f"Testing: max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion}")

                decision_tree = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
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
                        'criterion': criterion
                    }
                    best_accuracy = val_accuracy

    # evaluate best model on test set
    best_test_predictions = best_model.predict(X_test)
    best_test_accuracy = accuracy_score(y_test, best_test_predictions)
    print(f"Best model test Accuracy: {best_test_accuracy:.2f} with max_depth={best_params['max_depth']}, "
          f"min_samples_split={best_params['min_samples_split']}, criterion={best_params['criterion']}")

    # evaluate runner-up model on test set
    if runnerup_model is not None:
        runnerup_test_predictions = runnerup_model.predict(X_test)
        runnerup_test_predictions = accuracy_score(y_test, runnerup_test_predictions)
        print(f"Runner-up model test Accuracy: {runnerup_test_predictions:.2f} with max_depth={runnerup_params['max_depth']}, "
              f"min_samples_split={runnerup_params['min_samples_split']}, criterion={runnerup_params['criterion']}")


def main():
    # don't perform the grid search, we already know the best parameters.
    best_model = DecisionTreeClassifier(
        max_depth=2,
        min_samples_split=2,
        criterion='entropy',
        random_state=SEED
        )

    best_model.fit(X_train, y_train)

    features = [
        'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'REPORT_DOW',
        'REPORT_DOY', 'REPORT_HOUR', 'HOOD_158', 'LONGITUDE', 'LATITUDE',
        'AVG_AGE', 'POPULATION', 'INCOME', 'EMPLOYMENT_RATE',
        'PREMISES_TYPE_0', 'PREMISES_TYPE_1', 'PREMISES_TYPE_2', 'PREMISES_TYPE_3', 'PREMISES_TYPE_4'
    ]

    # plot the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(
        best_model,
        max_depth=2,
        feature_names=features,
        class_names=['low-risk', 'high-risk'],
        filled=True
    )
    plt.title("Top Two Levels of the Best Decision Tree")
    plt.show()


if __name__ == '__main__':
    main()