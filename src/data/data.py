import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FILE_PATH = 'toronto_crime_data.csv'
SEED = 311
np.random.seed(SEED)


def _load_csv(path):
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    
    df = pd.read_csv(path)
    
    return df


def load_data(path):
    df = _load_csv(path)
    
    shuffled_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    labels = shuffled_df.loc[:,"OFFENCE"].to_numpy()
    data = shuffled_df.drop(['OFFENCE'], axis=1).to_numpy()
    
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