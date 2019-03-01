from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import DataPreProcessor
from Evaluator import Evaluator
from LabelPredictorUtils import prepare_data


def main():
    data_train_file_path = 'Data/train.csv'
    data_test_file_path = 'Data/test_with_label.csv'
    data_types = {'Survived': 'Categorical',
                  'Pclass': 'Categorical',
                  'Name_Affiliation': 'Categorical',
                  'Last_Name': 'Text',
                  'First_Name': 'Text',
                  'Sex': 'Categorical',
                  'Age': 'Numerical',
                  'SibSp': 'Numerical',
                  'Parch': 'Numerical',
                  'Ticket_Code': 'Categorical',
                  'Ticket_Number': 'Numerical',
                  'Fare': 'Numerical',
                  'Cabin_Floor': 'Categorical',
                  'Cabin_Rooms': 'Text',
                  'Embarked': 'Categorical',
                  'Family_Members': 'Numerical',
                  'is_Traveling_Alone': 'Categorical',
                  'Fare_Per_Passenger': 'Categorical',
                  'is_Female_with_Children': 'Categorical'
                  }
    print('Loading train data...')
    train_data_df = DataPreProcessor.load_data(data_train_file_path)
    print('Loading test data...')
    test_data_df = DataPreProcessor.load_data(data_test_file_path)

    print('Cleaning and transforming train data...')
    cleaned_train_data_df = DataPreProcessor.clean_data(train_data_df, data_types)
    print('Cleaning and transforming test data...')
    cleaned_test_data_df = DataPreProcessor.clean_data(test_data_df, data_types)

    print('Saving cleaned train data to file...')
    cleaned_train_data_df.to_csv("clean_train.csv", index=False)
    print('Saving cleaned test data to file...')
    cleaned_test_data_df.to_csv("clean_test.csv", index=False)

    eval_classifiers = {
        'TreeClassifier': DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=4, random_state=42),
        'AdaBoost': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=42),
        'LogisticRegression': LogisticRegression(penalty='l1', max_iter=10000, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=2000, max_depth=4, min_samples_split=10, random_state=42, class_weight={0: 0.80, 1: 0.20}),
        'GBTrees': GradientBoostingClassifier(max_depth=4, learning_rate=0.1, n_estimators=2000, random_state=42, min_samples_split=10),
        'xgboost': XGBClassifier(max_depth=4, n_estimators=1000, random_state=42, learning_rate=0.1, min_samples_split=10),
        'KNN': KNeighborsClassifier(n_neighbors=3, p=2),
        'SVM': SVC(gamma='auto', tol=1e-3, C=1.5, random_state=42),
        'GBC': GradientBoostingClassifier(n_estimators=1000, max_depth=4, learning_rate=0.1)
    }

    eval_classifiers_params_grid = {
        'TreeClassifier': {'max_depth': [4, 5, 6]},
        'AdaBoost': {'n_estimators': [100, 200, 500, 1000, 2000],
                     'learning_rate': [0.2, 0.1, 0.05, 0.01]},
        'LogisticRegression': {'penalty': ['l1', 'l2']},
        'RandomForestClassifier': {'n_estimators': [100, 200, 500, 1000, 2000],
                                   'max_depth': [4, 5, 6],
                                   'max_features': [0.8, 0.5, 0.2, 0.1]},
        'GBTrees': {'n_estimators': [100, 500, 1000, 2000],
                    'max_depth': [4, 5, 6],
                    'max_features': [0.8, 0.5, 0.2, 0.1],
                    'learning_rate': [0.2, 0.1, 0.05, 0.01]},
        'xgboost': {'n_estimators': [100, 500, 1000, 2000],
                    'max_depth': [4, 5, 6],
                    'max_features': [0.8, 0.5, 0.2, 0.1],
                    'learning_rate': [0.2, 0.1, 0.05, 0.01]},
        'KNN': {'n_neighbors': [2, 3, 4, 5]},
        'SVM': {'gamma': [0.001, 0.01, 0.1, 1],
                'C': [1, 10, 50, 100, 200]},
        'GBC': {'n_estimators': [100, 500, 1000, 2000],
                'max_depth': [4, 5, 6, 8],
                'max_features': [0.8, 0.5, 0.2, 0.1],
                'learning_rate': [0.2, 0.1, 0.05, 0.01]}
    }

    # features_cols = ['Pclass', 'Sex', 'Age_Intevals', 'Family_Members', 'Fare_Per_Passenger', 'Embarked', 'Name_Affiliation', 'Ticket_Code', 'Cabin_Floor']
    # features_cols = ['Pclass', 'Sex', 'Age_Intevals', 'is_Traveling_Alone', 'Fare_Per_Passenger', 'Embarked', 'Name_Affiliation', 'Cabin_Floor']
    features_cols = ['Pclass', 'Sex', 'Age_Intervals', 'Name_Affiliation', 'Cabin_Floor', 'is_Female_with_Children']
    one_hot_encoding_features = ['Name_Affiliation', 'Cabin_Floor', 'Pclass']
    train_X, train_y = prepare_data(cleaned_train_data_df, class_col='Survived', features_cols=features_cols, one_hot_encoding_features=one_hot_encoding_features)
    test_X, test_y = prepare_data(cleaned_test_data_df, class_col='Survived', features_cols=features_cols, one_hot_encoding_features=one_hot_encoding_features)
    evaluator = Evaluator(train_X, train_y, test_X, test_y,  eval_classifiers, eval_classifiers_params_grid)
    # evaluator.select_features(selection_clf=ExtraTreesClassifier(n_estimators=1000, max_depth=4, random_state=42))

    all_predictions, final_prediction = evaluator.build_models(grid_search=False)
    evaluation_df = evaluator.save_predictions_to_df(all_predictions, final_prediction)
    submission_df = evaluator.save_predictions_for_submission(evaluation_df)
    evaluation_df.to_csv("test_evaluation_results.csv", index=False)
    submission_df.to_csv("test_submission.csv", index=False)
    accuracy = evaluator.evaluate_performance(test_y, final_prediction, performance_metric='accuracy')
    print('Accuracy for ensemble models {} is: {}'.format(eval_classifiers.keys(), accuracy))


if __name__ == '__main__':
    main()
