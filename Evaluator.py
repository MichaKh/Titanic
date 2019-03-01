import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score

from LabelPredictor import LabelPredictor


class Evaluator:
    eval_classifiers = {}
    eval_classifiers_params_grid = {}
    train_X = None
    train_y = None
    test_X = None
    test_y = None

    def __init__(self, train_X, train_y, test_X, test_y, eval_classifiers, eval_classifiers_params_grid):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.eval_classifiers = eval_classifiers
        self.eval_classifiers_params_grid = eval_classifiers_params_grid

    def select_features(self, selection_clf):
        sfm = SelectFromModel(selection_clf, threshold=0.25)
        sfm.fit(self.train_X, self.train_y)
        print(sfm.estimator_.feature_importances_)
        feature_idx = sfm.get_support()
        feature_names = self.train_X.columns[feature_idx]
        self.train_X = self.train_X.loc[:, list(feature_names)]
        self.test_X = self.test_X.loc[:, list(feature_names)]
        print(self.train_X.columns)

    def build_models(self, grid_search=False):
        all_predictions = {}
        for classifier in self.eval_classifiers:
            clf = self.eval_classifiers[classifier]
            predictor = LabelPredictor(self.train_X, self.train_y, clf)
            trained_clf = predictor.train_classifier(classifier_name=classifier,
                                                     classifier=clf,
                                                     grid_search=grid_search,
                                                     params_grid=self.eval_classifiers_params_grid[classifier])
            # Predict on test set
            test_predictions = predictor.predict_with_classifier(test_X=self.test_X, classifier_name=classifier,
                                                                 classifier=trained_clf)
            all_predictions[classifier + '_pred'] = test_predictions
            curr_model_performance = self.evaluate_performance(self.test_y, test_predictions)
            print("Aaccuracy of {} alone on test set:{}".format(classifier, curr_model_performance))

            # Predict on train set
            test_predictions = predictor.predict_with_classifier(test_X=self.train_X, classifier_name=classifier,
                                                                 classifier=trained_clf)
            curr_model_performance = self.evaluate_performance(self.train_y, test_predictions)
            print("Aaccuracy of {} alone on train set:{}".format(classifier, curr_model_performance))

            print('-'*68)
        final_prediction = self.get_ensemble_majority_vote(all_predictions)
        return all_predictions, final_prediction

    @staticmethod
    def get_ensemble_majority_vote(all_predictions):
        """
        Get majority vote of all predictions of built classifiers.
        :param all_predictions: Prediction vectors of all trained classifiers.
        :return: Series of final majority vote prediction
        """
        majority_vote_pred = []
        # zip all lists
        zipped_list = zip(*all_predictions.values())
        for l in zipped_list:
            maj_vote = max(l, key=l.count)
            majority_vote_pred.append(maj_vote)
        return pd.Series(majority_vote_pred)

    @staticmethod
    def evaluate_performance(actual_y, pred_y, performance_metric='accuracy'):
        """
        Evaluate the performance of predictions on a hold out set with known class labels.
        Three performance measures are supported: accuracy (default), f1-score and AUC.
        :param actual_y: Actual class label
        :param pred_y: Predicted class label
        :param performance_metric: One of three performance measures: accuracy, f1-score and AUC
        :return: Float value of performance
        """
        performance_metrics = {
            'accuracy': lambda actual, pred: accuracy_score(actual, pred, normalize=True),
            'f1': lambda actual, pred: f1_score(actual, pred, average='micro'),
            'auc': lambda actual, pred: metrics.auc(metrics.roc_curve(actual, pred, pos_label=1)[0],
                                                    metrics.roc_curve(actual, pred, pos_label=1)[1])
        }

        return performance_metrics[performance_metric](actual_y, pred_y)

    def save_predictions_to_df(self, all_predictions, final_prediction):
        eval_df = pd.DataFrame()
        eval_df = eval_df.append(self.test_X)
        eval_df['Survived'] = self.test_y
        for clf_pred in all_predictions:
            pred_col = all_predictions[clf_pred]
            eval_df[clf_pred] = pred_col.values
        eval_df['Survived_pred'] = final_prediction.values
        return eval_df

    @staticmethod
    def save_predictions_for_submission(eval_df):
        """
        Save predictions for submission in Kaggle.
        The submission consists of the PassangerId and whether he survived the accident.
        :param eval_df: Prediction Dataframe
        :return: Dataframe for submission
        """
        submission_df = pd.DataFrame()
        submission_df['PassengerId'] = eval_df.index
        submission_df['Survived'] = eval_df['Survived_pred'].values
        return submission_df
