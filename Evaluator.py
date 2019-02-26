from sklearn import metrics
import pandas as pd
from LabelPredictor import LabelPredictor
from sklearn.metrics import accuracy_score, f1_score


class Evaluator:
    eval_classifiers = {}
    train_X = None
    train_y = None
    test_X = None
    test_y = None

    def __init__(self, train_X, train_y, test_X, test_y, eval_classifiers):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.eval_classifiers = eval_classifiers

    def build_models(self):
        all_predictions = {}
        for classifier in self.eval_classifiers:
            clf = self.eval_classifiers[classifier]
            predictor = LabelPredictor(self.train_X, self.train_y, clf)
            trained_clf = predictor.train_classifier(classifier_name=classifier,
                                                     classifier=clf)
            predictions = predictor.predict_with_classifier(classifier_name=classifier,
                                                            classifier=trained_clf)
            all_predictions[classifier + '_pred'] = predictions
        final_prediction = self.get_ensemble_majority_vote(all_predictions)
        return all_predictions, final_prediction

    @staticmethod
    def get_ensemble_majority_vote(all_predictions):
        majority_vote_pred = []
        # zip all lists
        zipped_list = zip(*all_predictions)
        for l in zipped_list:
            maj_vote = max(l, key=l.count)
            majority_vote_pred.append(maj_vote)
        return pd.Series(majority_vote_pred)

    def evaluate_performance(self, pred_y, performance_metric='accuracy'):
        performance_metrics = {
            'accuracy': lambda actual, pred: accuracy_score(actual, pred, normalize=True),
            'f1': lambda actual, pred: f1_score(actual, pred, average='micro'),
            'auc': lambda actual, pred: metrics.auc(metrics.roc_curve(actual, pred, pos_label=2)[0],
                                                    metrics.roc_curve(actual, pred, pos_label=2)[1])
        }

        return performance_metrics[performance_metric](self.test_y, pred_y)

    def save_predictions_to_df(self, all_predictions, final_prediction):
        eval_df = pd.DataFrame()
        eval_df = eval_df.append(self.test_X)
        eval_df['Survived'] = self.test_y
        for clf_pred in all_predictions:
            eval_df[clf_pred] = all_predictions[clf_pred]
        eval_df['Survived_pred'] = final_prediction
        return eval_df

