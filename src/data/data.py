import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FILE_PATH = 'toronto_crime_data.csv'
SEED = 311
np.random.seed(SEED)


def _load_csv(path):
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    
    df = pd.read_csv(path)
    
    return df


def process_features(df):
    df['REPORT_DATE'] = pd.to_datetime(df['REPORT_DATE'], errors='coerce')
    df['Year'] = df['REPORT_DATE'].dt.year
    df['Month'] = df['REPORT_DATE'].dt.month
    df['Day'] = df['REPORT_DATE'].dt.day
    df['REPORT_DOW_NUM'] = df['REPORT_DOW'].map({
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
        'Friday': 5, 'Saturday': 6, 'Sunday': 7
    })

    # drop all rows with missing features
    df = df.dropna()

    # convert string-based/categorical features in to numbers
    # NOT one-hot encoding, but 0...n-1 encoding.
    # suitable Decision Trees/Random Forest/KDE.
    categorical_cols = ['PREMISES_TYPE', 'NBH_DESIGNATION', 'OFFENCE']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    features = [
        'REPORT_DOY', 'REPORT_HOUR', 'PREMISES_TYPE', 'HOOD_158',
        'AVG_AGE', 'POPULATION', 'INCOME', 'EMPLOYMENT_RATE',
        'Year', 'Month', 'Day', 'REPORT_DOW_NUM'
    ]
    target = 'OFFENCE'

    X = df[features].to_numpy()
    y = df[target].to_numpy()

    return X, y, label_encoders


def load_data(path):
    # shuffle-n-split
    df = _load_csv(path)
    data, labels, label_encoders = process_features(df)
    shuffled_indices = np.random.permutation(len(data))
    data, labels = data[shuffled_indices], labels[shuffled_indices]

    return split_data(data, labels)


def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    return X_train, y_train, X_val, y_val, X_test, y_test


# if __name__ == '__main__':
#     X_train, y_train, X_val, y_val, X_test, y_test = load_data(FILE_PATH)
#     print(X_train.shape)
#     print(X_test.shape)
#     print(X_val.shape)