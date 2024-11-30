import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score

from src.data import *

# Sample data function to randomly sample from the dataset
def sample_data(X, y, sample_size, random_seed=311):
    np.random.seed(random_seed)
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    return X[indices], y[indices]

# Grid search over bandwidth values for KDE
def grid_search_kde_bandwidth(X_train, y_train, classes, bandwidths):
    results = {}
    best_bandwidths = {}  # Dictionary to store the best bandwidth for each class

    for cls in classes:
        # Extract lat/long coordinates for the class
        X_cls = X_train[y_train == cls][:, [9, 10]]  # Assuming lat/long are columns 9 and 10
        log_likelihoods = []

        for bw in bandwidths:
            kde = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde.fit(X_cls)
            log_likelihood = kde.score(X_cls)  # Total log-likelihood
            log_likelihoods.append(log_likelihood)

        # Store results for each class
        results[cls] = {
            "bandwidths": bandwidths,
            "log_likelihoods": log_likelihoods
        }

        # Find the best bandwidth for the current class
        best_idx = np.argmax(log_likelihoods)
        best_bandwidth = bandwidths[best_idx]
        best_bandwidths[cls] = best_bandwidth

    return results, best_bandwidths

# Fit KDE with the best bandwidth for each class based on grid search results
def fit_best_kde(X_train, y_train, classes, results):
    best_kde_models = {}
    for cls in classes:
        # Find the bandwidth with the highest log-likelihood
        best_idx = np.argmax(results[cls]["log_likelihoods"])
        best_bandwidth = results[cls]["bandwidths"][best_idx]

        # Fit KDE with the best bandwidth
        X_cls = X_train[y_train == cls][:, [9, 10]]
        kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth)
        kde.fit(X_cls)
        best_kde_models[cls] = kde

    return best_kde_models

# Plot the results of the KDE grid search
def plot_kde_grid_search(results, bandwidths):
    plt.figure(figsize=(12, 8))
    for cls, data in results.items():
        plt.plot(
            bandwidths,
            data["log_likelihoods"],
            label=f'Class {cls}', marker='o'
        )

    plt.title('KDE Bandwidth Grid Search Results')
    plt.xlabel('Bandwidth')
    plt.ylabel('Log-Likelihood')
    plt.legend(title='Offense Class')
    plt.grid(True)
    plt.show()

# Find the best bandwidth for each class
def find_best_bandwidth_per_class(log_likelihoods, bandwidths):
    best_bandwidths = {}
    for cls, scores in log_likelihoods.items():
        max_idx = np.argmax(scores)
        best_bandwidths[cls] = bandwidths[max_idx]
    return best_bandwidths

# Find the global best bandwidth (based on the average log-likelihood)
def find_best_bandwidth_global(log_likelihoods, bandwidths):
    avg_log_likelihood = np.mean(np.array(list(log_likelihoods.values())), axis=0)
    best_idx = np.argmax(avg_log_likelihood)
    return bandwidths[best_idx]

# Predict using the KDE model
def predict_kde(kde_models, X, classes):
    predictions = []
    for sample in X:
        log_probs = []
        for cls in classes:
            log_prob = 0
            for feature_idx, kde in kde_models[cls].items():
                log_prob += kde.score_samples(sample[feature_idx].reshape(1, -1))
            log_probs.append(log_prob)
        predictions.append(np.argmax(log_probs))
    return np.array(predictions)

# Evaluate the accuracy of the KDE model
def evaluate_accuracy(kde_models, X, y, classes):
    y_pred = predict_kde(kde_models, X, classes)
    return accuracy_score(y, y_pred)

# Grid search for bandwidth and evaluation of accuracy
def grid_search_bandwidth_accuracy(X_train, y_train, X_val, y_val, classes, bandwidths):
    best_bandwidth = None
    best_accuracy = -1
    accuracies = []

    for bandwidth in bandwidths:
        # Fit KDE models with the current bandwidth
        kde_results = grid_search_kde_bandwidth(X_train, y_train, classes, bandwidths)
        best_kde_models = fit_best_kde(X_train, y_train, classes, kde_results)

        # Evaluate accuracy on validation data
        accuracy = evaluate_accuracy(best_kde_models, X_val, y_val, classes)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_bandwidth = bandwidth

    return best_bandwidth, best_accuracy, accuracies

# Function to plot bandwidth vs accuracy
def plot_bandwidth_vs_accuracy(bandwidths, accuracies):
    plt.figure(figsize=(12, 8))
    plt.plot(bandwidths, accuracies, marker='o')
    plt.title('Bandwidth vs Accuracy')
    plt.xlabel('Bandwidth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(FILE_PATH)
    sample_size = 2000
    # Sample the training and validation sets
    X_train_sample, y_train_sample = sample_data(X_train, y_train, sample_size)
    X_val_sample, y_val_sample = sample_data(X_val, y_val, sample_size)

    # Perform grid search for bandwidth
    bandwidths = np.linspace(0.01, 0.2, 20)  # Adjust the range as needed
    classes = np.unique(y_train)

    kde_results, best_bandwidths = grid_search_kde_bandwidth(X_train_sample, y_train_sample, classes, bandwidths)

    # Print the best bandwidth for each class
    print("Best bandwidths for each class:")
    for cls, best_bw in best_bandwidths.items():
        print(f"Class {cls}: Best Bandwidth = {best_bw}")
