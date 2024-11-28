import os
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# let FILE_PATH to be consistent across different function callers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'toronto_crime_data.csv')
SEED = 311
np.random.seed(SEED)

FEATURES = ['REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'REPORT_DOW',
            'REPORT_DOY', 'REPORT_HOUR', 'PREMISES_TYPE', 
            'HOOD_158', 'LONGITUDE', 'LATITUDE',
            'AVG_AGE', 'POPULATION', 'INCOME', 'EMPLOYMENT_RATE']
TARGET = 'OFFENCE'
TARGET_DIM = 2

def _load_csv(path):
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    df = pd.read_csv(path, low_memory=False)
    return df


def process_features(df):
    categorical_cols = ['PREMISES_TYPE', 'OFFENCE']

    # use labelencoder to convert categories/strings into integers
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    return X, y, label_encoders


def load_data():
    # shuffle-n-split
    df = _load_csv(FILE_PATH)
    data, labels, label_encoders = process_features(df)
    shuffled_indices = np.random.permutation(len(data))
    data, labels = data[shuffled_indices], labels[shuffled_indices]

    return _split_data(data, labels)


def _split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    return X_train, y_train, X_val, y_val, X_test, y_test

def np_to_tensor(data:list[np.array]):
    return [torch.from_numpy(d.astype(np.float64)) for d in data]

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(np.unique(y_test))
