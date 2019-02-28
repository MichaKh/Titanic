from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LassoCV
from sklearn.naive_bayes import MultinomialNB
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
                  'Embarked': 'Categorical'}
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
        'LogisticRegression': LogisticRegression(penalty='l1', max_iter=1000, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=42),
        'GBTrees': GradientBoostingClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, random_state=42, min_samples_split=2),
        'xgboost': XGBClassifier(max_depth=6, n_estimators=1000, random_state=42, learning_rate=0.1, min_samples_split=2),
        'KNN': KNeighborsClassifier(n_neighbors=3, p=2),
        'SVM': SVC(gamma='auto', tol=1e-3, C=1.5, random_state=42),
    }
    # features_cols = ['Pclass', 'Sex', 'Age_Intevals', 'Family_Members', 'Fare_Per_Passenger', 'Embarked', 'Name_Affiliation', 'Ticket_Code', 'Cabin_Floor']
    features_cols = ['Pclass', 'Sex', 'Age_Intevals', 'is_Traveling_Alone', 'Fare_Per_Passenger', 'Embarked', 'Name_Affiliation', 'Cabin_Floor']
    train_X, train_y = prepare_data(cleaned_train_data_df, class_col='Survived', features_cols= features_cols)
    test_X, test_y = prepare_data(cleaned_test_data_df, class_col='Survived', features_cols=features_cols)
    evaluator = Evaluator(train_X, train_y, test_X, test_y,  eval_classifiers)
    # evaluator.select_features(selection_clf=ExtraTreesClassifier(n_estimators=1000, max_depth=4, random_state=42))

    all_predictions, final_prediction = evaluator.build_models()
    evaluation_df = evaluator.save_predictions_to_df(all_predictions, final_prediction)
    submission_df = evaluator.save_predictions_for_submission(evaluation_df)
    evaluation_df.to_csv("test_evaluation_results.csv", index=False)
    submission_df.to_csv("test_submission.csv", index=False)
    accuracy = evaluator.evaluate_performance(test_y, final_prediction, performance_metric='accuracy')
    print('Accuracy for ensemble models {} is: {}'.format(eval_classifiers.keys(), accuracy))


if __name__ == '__main__':
    main()
