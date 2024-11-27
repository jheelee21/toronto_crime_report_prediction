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


def compute_mle_prior(y_train, classes):
    """Compute weighted priors inversely proportional to class frequencies."""
    class_counts = Counter(y_train)
    class_weights = {cls: 1 / count for cls, count in class_counts.items()}
    total_weight = sum(class_weights.values())
    weighted_priors = {cls: weight / total_weight for cls, weight in class_weights.items()}
    return weighted_priors


def fit_kde_independent(X_train, y_train, classes):
    """Apply KDE to each feature independently."""
    kde_models = {}
    for cls in classes:
        kde_models[cls] = {}
        X_cls = X_train[y_train == cls]
        for feature_idx in range(X_cls.shape[1]):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.001)
            kde.fit(X_cls[:, feature_idx].reshape(-1, 1))
            kde_models[cls][feature_idx] = kde
    return kde_models


def fit_geo_kde(X_train, y_train, classes):
    """Apply KDE to geospatial data (longitude and latitude) together as interdependent features."""
    kde_models = {}
    for cls in classes:
        X_cls = X_train[y_train == cls][:, [9, 10]]  # Extract only LONG_WGS84 and LAT_WGS84
        kde = KernelDensity(kernel='gaussian', bandwidth=0.0001)
        kde.fit(X_cls)
        kde_models[cls] = kde
    return kde_models


def fit_time_kde(X_train, y_train, classes, time_features_columns=[0, 1, 2, 3, 4, 5], bandwidth=0.001):
    kde_models = {}
    for cls in classes:
        kde_models[cls] = {}
        # Use boolean indexing to extract rows for the given class
        X_cls = X_train[y_train == cls]  # This will give all features for the given class
        X_cls_time = X_cls[:, time_features_columns]  # Now select only the time-related columns
        for feature_idx in range(X_cls_time.shape[1]):
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(X_cls_time[:, feature_idx].reshape(-1, 1))  # Fit KDE to the time feature
            kde_models[cls][feature_idx] = kde
    return kde_models


def compute_geo_likelihood(kde_models, X, classes):
    """Compute the likelihood for geospatial features (longitude and latitude)."""

    # Initialize an array to store geo likelihood for each sample and class
    geo_likelihood = np.zeros((X.shape[0], len(classes)))  # X.shape[0] is the number of samples

    # For each class, compute the likelihood using KDE on the last two features (longitude, latitude)
    for i, cls in enumerate(classes):
        # Using the last two columns (longitude, latitude)
        geo_likelihood[:, i] = kde_models[cls].score_samples(X[:, [9, 10]])

    return geo_likelihood


def compute_time_likelihood(kde_models, X, classes, time_features_columns=[0, 1, 2, 3, 4, 5]):

    time_likelihood = np.zeros((X.shape[0], len(classes)))

    for i, cls in enumerate(classes):
        # Extract only the time features from X
        X_time = X[:, time_features_columns]
        # Initialize likelihood for this class
        likelihood = 0
        for feature_idx in range(X_time.shape[1]):
            # Get the KDE model for this feature and class
            kde = kde_models[cls][feature_idx]
            # Calculate the log likelihood for each time feature and add them together
            likelihood += kde.score_samples(X_time[:, feature_idx].reshape(-1, 1))
        # Store the combined likelihood for this class
        time_likelihood[:, i] = likelihood

    return time_likelihood


def compute_likelihood_independent(kde_models, X, cls):
    """Compute likelihood for each feature independently."""
    likelihood = 1
    for feature_idx in range(X.shape[1]):  # Loop over all features
        kde = kde_models[cls][feature_idx]
        log_likelihood = kde.score_samples(X[:, feature_idx].reshape(-1, 1))
        likelihood *= np.exp(log_likelihood)  # Convert from log-likelihood
    return likelihood


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
                likelihood = likelihood_geo + likelihood_time


            posterior = likelihood * priors[cls]
            posteriors[cls] = posterior
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
        kde_models = [kde_models_geo, kde_models_time]

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
    scaler = MinMaxScaler()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(FILE_PATH)
    # Fit scaler on training data for longitude and latitude (columns [9, 10])
    X_train[:, [9, 10]] = scaler.fit_transform(X_train[:, [9, 10]])
    X_test[:, [9, 10]] = scaler.transform(X_test[:, [9, 10]])

    sample_size = 2000
    # Sample the training and validation sets
    X_train_sample, y_train_sample = sample_data(X_train, y_train, sample_size)
    X_val_sample, y_val_sample = sample_data(X_val, y_val, sample_size)
    X_test_sample, y_test_sample = sample_data(X_test, y_test, sample_size)

    evaluate_models(X_train_sample, y_train_sample, X_test, y_test)

    # classes = np.unique(y_train)
    # features = X_train.shape[1]
    #
    # # Fit KDE models
    # kde_models = fit_kde(X_train, y_train, classes, features)
    #
    # # Compute priors
    # priors_mle = compute_mle_prior(y_train, classes)
    #
    # predictions_mle = predict(kde_models, priors_mle, X_val, classes)
    #
    #
    # accuracy_mle = np.mean(predictions_mle == y_val)
    #
    # print(f"Accuracy using MLE: {accuracy_mle}")

    # evaluate_models(X_train, y_train, X_test, y_test, bandwidth=0.1)