from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


def prepare_data(train_data_df, class_col, features_cols, one_hot_encoding_features=[]):
    """
    Split the data to features and label column, i.e., X and y.
    Encode the categorical features to numeric encoding (to fit "sklearn" implementations)
    Only provided features will be encoded as one-hot-vectors, rest are encoding with regular categorical encoding
    :param one_hot_encoding_features: List of features to encode to one-hot vectors
    :param train_data_df: Input data
    :param class_col: Column name representing the class label
    :param features_cols: Column names representing the data features
    :return: Dataframes of X and y.
    """
    data_X, data_y = split_features_and_label(train_data_df, class_col, features_cols)

    if one_hot_encoding_features:
        reg_encoding_cols = data_X.loc[:, [f not in one_hot_encoding_features for f in features_cols]]
        data_X_reg_encoding = encode_features(reg_encoding_cols)

        one_hot_data_X = data_X.loc[:, one_hot_encoding_features]
        data_X_one_hot_encoded = encode_one_hot_features(one_hot_data_X)
        data_X = pd.concat([data_X_reg_encoding, data_X_one_hot_encoded], axis=1)
    else:
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
        if train_X[col].dtype not in [np.float64, np.int64, np.int32, np.float32]:  # not numeric
            train_X.loc[:, col] = le.fit_transform(train_X[col])
    return train_X


def encode_one_hot_features(train_X):
    encoded_train_X = pd.DataFrame()
    for col in train_X.columns:
        if train_X[col].dtype not in [np.float64, np.int64, np.int32, np.float32]:  # not numeric
            encoded = pd.get_dummies(train_X[col], prefix=col, drop_first=False)
            encoded_train_X = pd.concat([encoded_train_X, encoded], axis=1)
        else:
            encoded_train_X = pd.concat([encoded_train_X, train_X[col]], axis=1)
    return encoded_train_X