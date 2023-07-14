import lightgbm as lgb
import os.path as osp
import numpy as np
import json
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from utils.logger import get_logger

from joblib import dump,load

from sklearn.neural_network import MLPClassifier


class MLP:
    def __init__(self, args):
        self.logger = get_logger(args.log_dir, args.model)
        self.args = args

        mlp_params = {
            'hidden_layer_sizes': (args.hidden_layer_sizes,100),
            'max_iter': args.max_iter,
            'random_state': args.random_state,
            'verbose': True
        }

        self.logger.info("MLP params: {}"
                         .format(json.dumps(mlp_params)))
        self.mlp_clf = MLPClassifier(**mlp_params)

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
        self.logger.info("=================Start mlp training phase ...=================")

        X, y, sample_ids = train_dataset
        # training
        self.mlp_clf.fit(X, y)
        self.predict_eval(X, y, self.mlp_clf)

    def evaluate(self, test_dataset):
        self.logger.info("=========================Start mlp evaluating phase ...==============================")

        # load mlp model
        mlp_clf = self.mlp_clf
        X_valid, y_valid, sample_ids = test_dataset
        self.predict_eval(X_valid, y_valid, mlp_clf)