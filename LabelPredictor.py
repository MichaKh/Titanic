import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class LabelPredictor:
    eval_classifier = None
    train_X = None
    train_y = None

    def __init__(self, train_X, train_y, eval_classifier):
        self.eval_classifier = eval_classifier
        self.train_X = train_X
        self.train_y = train_y

    def train_classifier(self, classifier_name, classifier, grid_search, params_grid=None):
        print('Training classifier: {}'.format(classifier_name))
        start_time = time.time()
        if classifier is None:
            return None
        else:
            if grid_search and params_grid:
                grid_classifier = GridSearchCV(classifier, param_grid=params_grid, scoring='accuracy', n_jobs=4, verbose=1, cv=StratifiedKFold(n_splits=5))
                grid_classifier.fit(self.train_X, self.train_y)
                classifier = grid_classifier.best_estimator_
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

    @staticmethod
    def get_grid_search_type(classifier_name):
        ensem_param_grid = {'n_estimators': [100, 200, 500, 1000, 2000],
                            'max_depth': [4, 5, 6, 8],
                            'max_features': [0.8, 0.5, 0.2, 0.1]}
        if 'forest' in classifier_name.lower():
            return ensem_param_grid
        elif 'gb' in classifier_name.lower():
            ensem_param_grid['learning_rate'] = [0.2, 0.1, 0.05, 0.01]
        return ensem_param_grid
