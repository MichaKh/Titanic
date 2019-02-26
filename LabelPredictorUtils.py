from sklearn.preprocessing import LabelEncoder
import numpy as np


def prepare_data(train_data_df, class_col, features_cols):
    data_X, data_y = split_features_and_label(train_data_df, class_col, features_cols)
    data_X = encode_features(data_X)
    return data_X, data_y


def split_features_and_label(train_data_df, class_col, features_cols):
    train_y = train_data_df[class_col]
    train_X = train_data_df.loc[:, train_data_df.columns.isin(features_cols)]
    assert len(train_y) == train_X.shape[0]
    return train_X, train_y


def encode_features(train_X):
    le = LabelEncoder()
    for col in train_X.columns:
        if train_X[col].dtype not in [np.float64, np.int64]:  # numeric
            train_X[col] = le.fit_transform(train_X[col])
    return train_X