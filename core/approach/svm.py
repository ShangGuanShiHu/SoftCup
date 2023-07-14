import lightgbm as lgb
import os.path as osp
import numpy as np
import json
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from utils.logger import get_logger
import multiprocessing as mp

from sklearn.svm import SVC
from joblib import dump,load
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self, args):
        self.logger = get_logger(args.log_dir, args.model)
        self.args = args

        svm_params = {
            'C': args.C,
            'kernel': args.kernel,
            'decision_function_shape': args.decision_function_shape
        }

        self.logger.info("SVC params: {}"
                         .format(json.dumps(svm_params)))

        self.svm_clf = SVC(**svm_params)

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
        self.logger.info("=================Start svm training phase ...=================")

        X, y, sample_ids = train_dataset
        # training
        self.svm_clf.fit(X, y)

        self.predict_eval(X, y, self.svm_clf)

    def evaluate(self, test_dataset):
        self.logger.info("=========================Start svm evaluating phase ...==============================")

        # load svm model
        svm_clf = self.svm_clf
        X_valid, y_valid, sample_ids = test_dataset
        self.predict_eval(X_valid, y_valid, svm_clf)