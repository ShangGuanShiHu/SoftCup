from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

from utils.logger import get_logger


class CatBoost:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger(args.log_dir, args.model)
        self.params = {
            'iterations': 1000,
            'learning_rate': 0.4,
            'depth': 3,
            'loss_function': 'MultiClass'
        }

    def fit(self, train_dataset, test_dataset):
        X, y, sample_ids = train_dataset
        clf = CatBoostClassifier(**self.params)
        clf.fit(X, y)
        pred_labels = clf.predict(X)
        self.logger.debug("\n{}".format(
            classification_report(y, pred_labels, digits=4)
        ))
        pred_labels = clf.predict(test_dataset[0])
        self.logger.debug("\n{}".format(
            classification_report(test_dataset[1], pred_labels, digits=4)
        ))