import lightgbm as lgb
import os.path as osp
import numpy as np
import json
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from utils.logger import get_logger
import multiprocessing as mp


class RandomForest:
    def __init__(self, args):
        self.logger = get_logger(args.log_dir, args.model)
        self.args = args

        randomforest_params = {
            'n_estimators' : 100,
            'random_state' : 42
        }

        self.logger.info("randomforest params: {}"
                         .format(json.dumps(randomforest_params)))

        self.randomforest_clf = RandomForestClassifier(**randomforest_params)

    def predict_eval(self, X, y, clf):
        pred_probs = clf.predict(X)
        pred_labels = pred_probs
        f1 = f1_score(y, pred_labels, average='macro')
        pre = precision_score(y, pred_labels, average='macro')
        rec = recall_score(y, pred_labels, average='macro')
        self.logger.debug("F1, Precison, Recall: {:.3%} {:.3%} {:.3%}".format(f1, pre, rec))
        self.logger.debug("F1 : {}".format(f1_score(y, pred_labels, average=None)))
        self.logger.debug("pre : {}".format(precision_score(y, pred_labels, average=None)))
        self.logger.debug("rec : {}".format(recall_score(y, pred_labels, average=None)))

    def fit(self, train_dataset, test_dataset=None):
        self.logger.info("=================Start randomforest_clf training phase ...=================")

        X, y, sample_ids = train_dataset
        # training
        self.randomforest_clf.fit(X, y)

        self.predict_eval(X, y, self.randomforest_clf)

    def evaluate(self, test_dataset):
        self.logger.info("=========================Start randomforest_clf evaluating phase ...==============================")

        # load randomforest_clf model
        randomforest_clf = self.randomforest_clf
        X_valid, y_valid, sample_ids = test_dataset
        self.predict_eval(X_valid, y_valid, randomforest_clf)