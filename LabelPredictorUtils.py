from sklearn.preprocessing import LabelEncoder
import numpy as np


def prepare_data(train_data_df, class_col, features_cols):
    """
    Split the data to features and label column, i.e., X and y.
    Encode the categorical feautres to numeric encoding (to fit "sklearn" implementations)
    :param train_data_df: Input data
    :param class_col: Column name representing the class label
    :param features_cols: Column names representing the data features
    :return: Dataframes of X and y.
    """
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
        if train_X[col].dtype not in [np.float64, np.int64, np.int32, np.float32]:  # not numeric
            train_X.loc[:, col] = le.fit_transform(train_X[col])
    return train_X