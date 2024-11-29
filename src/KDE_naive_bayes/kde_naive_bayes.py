from numba.cuda import runtime
from sklearn.preprocessing import MinMaxScaler
from KDE_naive_bayes.reduced_sample_test import sample_data
from src.data import *
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KernelDensity
from collections import Counter


def kde_with_silverman_bandwidth(X_train):
    bandwidth = 1.06 * np.std(X_train) * len(X_train) ** (-1 / 5)
    return bandwidth


def undersample_minority_class(X_train, y_train):
    """Repeat sampling of the minority class to balance the dataset."""
    np.random.seed(311)  # For reproducibility

    class_counts = Counter(y_train)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # Extract majority and minority class data
    X_majority = X_train[y_train == majority_class]
    X_minority = X_train[y_train == minority_class]

    # Compute how many samples are needed to balance
    majority_class_sampled = X_majority[
        np.random.choice(X_majority.shape[0], len(minority_class), replace=False)]

    X_undersampled = np.vstack((majority_class_sampled, X_minority))
    y_undersampled = np.hstack((np.zeros(len(minority_class)), np.ones(len(minority_class))))

    return X_undersampled, y_undersampled


def compute_mle_prior(y_train, classes):
    """Compute weighted priors inversely proportional to class frequencies."""
    class_counts = Counter(y_train)
    class_weights = {cls: 1 / count for cls, count in class_counts.items()}
    total_weight = sum(class_weights.values())
    weighted_priors = {cls: max(class_counts.values()) / total_weight for cls, weight in class_weights.items()}
    return weighted_priors


def fit_kde_independent(X_train, y_train, classes, feature_range=None, bandwidth=0.01):
    """Apply KDE to each feature independently."""
    kde_models = {}

    if feature_range is None:
        for cls in classes:
            kde_models[cls] = {}
            # Filter the rows of X_train for the current class
            X_cls = X_train[y_train == cls]
            optimal_bandwidth = kde_with_silverman_bandwidth(X_cls)
            for feature_idx in range(X_train.shape[1]):
                feature_data = X_cls[:, feature_idx]
                kde = KernelDensity(kernel='gaussian', bandwidth=optimal_bandwidth)
                kde.fit(feature_data.reshape(-1, 1))
                kde_models[cls][feature_idx] = kde
    else:
        for cls in classes:
            kde_models[cls] = {}
            # Filter the rows of X_train for the current class
            X_cls = X_train[y_train == cls]
            optimal_bandwidth = kde_with_silverman_bandwidth(X_cls)
            for feature_idx in feature_range:
                feature_data = X_cls[:, feature_idx]
                kde = KernelDensity(kernel='gaussian', bandwidth=optimal_bandwidth)
                kde.fit(feature_data.reshape(-1, 1))
                kde_models[cls][feature_idx] = kde

    return kde_models


def fit_geo_kde(X_train, y_train, classes, bandwidth=0.0001):
    """Apply KDE to geospatial data (longitude and latitude) together as interdependent features."""
    kde_models = {}
    for cls in classes:
        X_cls = X_train[y_train == cls][:, [8, 9]]  # Extract only LONG_WGS84 and LAT_WGS84
        # optimal_bandwidth = kde_with_silverman_bandwidth(X_cls)  # Get optimal bandwidth
        kde = KernelDensity(kernel='gaussian', bandwidth=0.04)
        kde.fit(X_cls)
        kde_models[cls] = kde
    return kde_models


def fit_time_kde(X_train, y_train, classes, time_features_columns=[0, 1, 2, 3, 4, 5], bandwidth=0.001):
    kde_models = {}
    for cls in classes:
        kde_models[cls] = {}
        # Use boolean indexing to extract rows for the given class
        X_cls = X_train[y_train == cls][:, time_features_columns]  # This will give all features for the given class
        optimal_bandwidth = kde_with_silverman_bandwidth(X_cls)
        kde = KernelDensity(kernel='gaussian', bandwidth=optimal_bandwidth)
        kde.fit(X_cls)
        kde_models[cls] = kde
    return kde_models


def compute_geo_likelihood(kde_models, X, classes):
    """Compute the likelihood for geospatial features (longitude and latitude)."""

    # Initialize an array to store geo likelihood for each sample and class
    geo_likelihood = np.zeros((X.shape[0], len(classes)))  # X.shape[0] is the number of samples

    # For each class, compute the likelihood using KDE on the last two features (longitude, latitude)
    for i, cls in enumerate(classes):
        # Using the last two columns (longitude, latitude)
        geo_likelihood[:, i] = kde_models[cls].score_samples(X[:, [8, 9]])

    return np.clip(geo_likelihood, 1e-10, None)


def compute_time_likelihood(kde_models, X, classes, time_features_columns=[0, 1, 2, 3, 4, 5]):

    time_likelihood = np.zeros((X.shape[0], len(classes)))

    for i, cls in enumerate(classes):

        time_likelihood[:, i] = kde_models[cls].score_samples(X[:, time_features_columns])

    return np.clip(time_likelihood, 1e-10, None)


def compute_likelihood_independent(kde_models, X, cls, feature_range=None):
    """Compute likelihood for each feature independently."""
    likelihood = 1
    if feature_range is None:
        for feature_idx in range(X.shape[1]):  # Loop over all features
            kde = kde_models[cls][feature_idx]
            log_likelihood = kde.score_samples(X[:, feature_idx].reshape(-1, 1))
            likelihood *= np.exp(log_likelihood)  # Convert from log-likelihood
    else:
        for feature_idx in feature_range:  # Loop over all features
            kde = kde_models[cls][feature_idx]
            log_likelihood = kde.score_samples(X[:, feature_idx].reshape(-1, 1))
            likelihood *= np.exp(log_likelihood)
    return np.clip(likelihood, 1e-10, None)


def train_naive_bayes(X_train, y_train):
    """Train a Naive Bayes classifier."""

    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_priors = {cls: count / total_samples for cls, count in class_counts.items()}

    model = GaussianNB(priors=list(class_priors.values()))
    model.fit(X_train, y_train)
    return model


def predict(kde_models, priors, X, classes, model_type='geo'):
    """Make predictions using Naive Bayes with either geo-based or feature-based likelihoods."""
    predictions = []
    for x in X:
        posteriors = {}
        for cls in classes:
            if model_type == 'geo':
                likelihood = compute_geo_likelihood(kde_models, x.reshape(1, -1), [cls])
                likelihood = likelihood[0, 0]
            elif model_type == 'independent':
                likelihood = compute_likelihood_independent(kde_models, x.reshape(1, -1), cls)
            elif model_type == 'time':
                likelihood = compute_time_likelihood(kde_models, x.reshape(1, -1), [cls])
                likelihood = likelihood[0, 0]
            elif model_type == 'all':
                likelihood_geo = compute_geo_likelihood(kde_models[0], x.reshape(1, -1), [cls])[0, 0]
                likelihood_time = compute_time_likelihood(kde_models[1], x.reshape(1, -1), [cls])[0, 0]
                columns_left = [6, 7, 10, 11, 12, 13]
                likelihood_indp = compute_likelihood_independent(kde_models[2], x.reshape(1, -1), cls, columns_left)
                likelihood = likelihood_geo + likelihood_time + likelihood_indp


            posterior = likelihood * priors[cls]
            posteriors[cls] = posterior
        posteriors = {cls: posterior / sum(posteriors.values()) for cls, posterior in posteriors.items()}
        predictions.append(max(posteriors, key=posteriors.get))

    return np.array(predictions)


def kde_naive_bayes(X_train, y_train, X_test, classes, priors, model_type='geo'):
    """Apply Naive Bayes with KDE (either on geo or independent features)."""
    if model_type == 'geo':
        kde_models = fit_geo_kde(X_train, y_train, classes)
    elif model_type == 'independent':
        kde_models = fit_kde_independent(X_train, y_train, classes)
    elif model_type == 'time':
        kde_models = fit_time_kde(X_train, y_train, classes)
    elif model_type == 'all':
        kde_models_geo = fit_geo_kde(X_train, y_train, classes)
        kde_models_time = fit_time_kde(X_train, y_train, classes)
        columns_left = [6, 7, 10, 11, 12, 13]
        kde_model_indp = fit_kde_independent(X_train, y_train, classes, columns_left)
        kde_models = [kde_models_geo, kde_models_time, kde_model_indp]

    # Make predictions
    predictions = predict(kde_models, priors, X_test, classes, model_type=model_type)
    return predictions


def evaluate_models(X_train, y_train, X_test, y_test):
    """Evaluate the performance of both Naive Bayes models."""
    classes = np.unique(y_train)

    # Compute priors using MLE
    priors = compute_mle_prior(y_train, classes)

    start_time = time.time()
    # Direct Naive Bayes (without KDE)
    nb_model = train_naive_bayes(X_train, y_train)
    direct_preds = nb_model.predict(X_test)
    direct_acc = accuracy_score(y_test, direct_preds)

    # KDE with Naive Bayes (independent feature-based KDE)
    independent_preds = kde_naive_bayes(X_train, y_train, X_test, classes, priors, model_type='independent')
    independent_acc = accuracy_score(y_test, independent_preds)

    # KDE with Naive Bayes (geo-based KDE)
    geo_preds = kde_naive_bayes(X_train, y_train, X_test, classes, priors, model_type='geo')
    geo_acc = accuracy_score(y_test, geo_preds)

    # KDE with Naive Bayes (time-based KDE)
    time_preds = kde_naive_bayes(X_train, y_train, X_test, classes, priors, model_type='time')
    time_acc = accuracy_score(y_test, time_preds)

    # KDE geo + time
    all_preds = kde_naive_bayes(X_train, y_train, X_test, classes, priors, model_type='all')
    all_acc = accuracy_score(y_test, all_preds)

    end_time = time.time()
    _runtime = end_time - start_time
    print(f'Runtime: {_runtime:.4f} seconds.')
    # Print results
    print("Direct Naive Bayes Accuracy:", direct_acc)
    print("KDE-Enhanced Naive Bayes (Independent) Accuracy:", independent_acc)
    print("KDE-Enhanced Naive Bayes (Geo) Accuracy:", geo_acc)
    print("KDE-Enhanced Naive Bayes (Time) Accuracy:", time_acc)
    print("KDE-Enhanced Naive Bayes (All) Accuracy:", all_acc)


    print("\nClassification Report (Direct Naive Bayes):\n", classification_report(y_test, direct_preds))
    print("\nClassification Report (KDE-Enhanced Naive Bayes - Independent):\n",
          classification_report(y_test, independent_preds))
    print("\nClassification Report (KDE-Enhanced Naive Bayes - Geo):\n", classification_report(y_test, geo_preds))
    print("\nClassification Report (KDE-Enhanced Naive Bayes - Time):\n", classification_report(y_test, time_preds))
    print("\nClassification Report (KDE-Enhanced Naive Bayes - All):\n", classification_report(y_test, all_preds))


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Fit scaler on training data for longitude and latitude (columns [9, 10])
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    X_train_balanced, y_train_balanced = undersample_minority_class(X_train, y_train)

    # sample_size = 4000
    # # Sample the training and validation sets
    # X_train_sample, y_train_sample = sample_data(X_train_balanced, y_train_balanced, sample_size)
    # X_val_sample, y_val_sample = sample_data(X_val, y_val, sample_size)
    # X_test_sample, y_test_sample = sample_data(X_test, y_test, sample_size)

    # evaluate_models(X_train, y_train, X_test, y_test)
    evaluate_models(X_train_balanced, y_train_balanced, X_test, y_test)
