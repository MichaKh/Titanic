import time
import pandas as pd


class LabelPredictor:
    eval_classifier = None
    train_X = None
    train_y = None

    def __init__(self, train_X, train_y, eval_classifier):
        self.eval_classifier = eval_classifier
        self.train_X = train_X
        self.train_y = train_y

    def train_classifier(self, classifier_name, classifier):
        print('Training classifier: {}'.format(classifier_name))
        start_time = time.time()
        if classifier is None:
            return None
        else:
            classifier.fit(self.train_X, self.train_y)
            print("Training {} classifier completed in {} seconds".format(classifier_name, (time.time() - start_time)))
        return classifier

    @staticmethod
    def predict_with_classifier(test_X, classifier_name, classifier):
        print('Predicting with classifier: {}'.format(classifier_name))
        start_time = time.time()
        predictions = classifier.predict(test_X)
        print("Predicting with {} classifier completed in {} seconds".format(classifier_name, (time.time() - start_time)))
        return pd.Series(predictions)
