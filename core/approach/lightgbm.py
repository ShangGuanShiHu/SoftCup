import graphviz
import lightgbm as lgb
import os.path as osp
import numpy as np
import json

import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from utils.logger import get_logger
import multiprocessing as mp

# 在环境变量中加入安装的Graphviz路径
import os
os.environ["PATH"] += os.pathsep + 'F:/Graphviz/bin'


# 自定义评估函数
def custom_eval(y_true, y_pred):
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'f1_score', f1_score(y_true, y_pred, average='macro'), True


class LightGBM:
    def __init__(self, args):
        self.logger = get_logger(args.log_dir, args.model)
        self.args = args
        lambda_l1 = 0 if args.valid else 0.005
        lgb_params = {
            'boosting_type': args.boosting_type,
            'objective': args.objective,
            'num_iterations': args.num_iterations,
            'num_leaves': args.num_leaves,
            'max_depth': args.max_depth,
            'learning_rate': args.lgb_learning_rate,
            'n_jobs': mp.cpu_count(),
            'min_data_in_leaf': args.min_data_in_leaf,
            'early_stopping_rounds': args.early_stopping_rounds,
            'metric': ['custom'],
            'reg_lambda': lambda_l1,
            # 'feature_fraction': 0.9,
        }

        self.logger.info("LightGBM params: {}"
                         .format(json.dumps(lgb_params)))

        self.lgb_clf = lgb.LGBMClassifier(**lgb_params)


    def predict_eval(self, X, y, clf):
        pred_probs = clf.predict(X, num_iteration=clf.best_iteration_ + 1)
        # pred_probs = clf.predict(X)
        pred_labels = pred_probs
        self.logger.debug("\n{}".format(
            classification_report(y, pred_labels, digits=4)
        ))
        # f1 = f1_score(y, pred_labels, average='macro')
        # pre = precision_score(y, pred_labels, average='macro')
        # rec = recall_score(y, pred_labels, average='macro')
        # self.logger.debug("F1, Precison, Recall: {:.3%} {:.3%} {:.3%}".format(f1, pre, rec))
        # self.logger.debug("F1 : {}".format(f1_score(y, pred_labels, average=None)))
        # self.logger.debug("pre : {}".format(precision_score(y, pred_labels, average=None)))
        # self.logger.debug("rec : {}".format(recall_score(y, pred_labels, average=None)))
        # mistakes = {}
        # rights = {}
        # for i, label in enumerate(y):
        #     x = X[i]
        #     pred_label = pred_labels[i]
        #     if pred_labels[i] != label:
        #         label = f"{label}-{pred_label}"
        #         if label in mistakes.keys():
        #             mistakes[label] = np.concatenate([mistakes[label], x.reshape(1,len(x))], 0)
        #         else:
        #             mistakes[label] = x.reshape(1,len(x))
        #     else:
        #         if label in rights.keys():
        #             rights[label] = np.concatenate([rights[label], x.reshape(1,len(x))], 0)
        #         else:
        #             rights[label] = x.reshape(1,len(x))
        # for right in rights:
        #     df = DataFrame(rights[right])
        #     # print()
        # for mistake in mistakes:
        #     df = DataFrame(mistakes[mistake])
        #     # print()


    def fit(self, train_dataset, test_dataset=None, is_grid=False):
        self.logger.info("=================Start lightGBM training phase ...=================")

        X, y, sample_ids = train_dataset
        # sample_weight = []
        # for type in y:
        #     if type == 1 or type == 2:
        #         sample_weight.append(2.0)
        #     elif type == 0:
        #         sample_weight.append(0.8)
        #     else:
        #         sample_weight.append(0.2)
        if test_dataset is None:
            # self.lgb_clf.fit(X, y, sample_weight=sample_weight)
            self.lgb_clf.fit(X, y)
        else:
            X_valid, y_valid,_ = test_dataset
            # X = np.concatenate((X, X_valid), axis=0)
            # y = np.concatenate((y, y_valid), axis=0)
            # self.lgb_clf.fit(X, y
            #                  , eval_set=[(X_valid, y_valid)], eval_metric=custom_eval
            #                  , sample_weight=sample_weight)
            self.lgb_clf.fit(X, y
                             , eval_set=[(X_valid, y_valid)], eval_metric=custom_eval)
        # training
        self.lgb_clf.booster_.save_model(
            osp.join(self.args.model_dir, 'lgb')
        )

        # 获取特征的重要性分数
        feature_importance = self.lgb_clf.feature_importances_
        feature_index_importance = sorted(enumerate(feature_importance), key=lambda x: x[1] * -1)
        # train_df = DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[-1])])
        # train_df['label'] = y
        # test_df = DataFrame(X_valid, columns=[f'feature_{i}' for i in range(X.shape[-1])])
        # test_df['label'] = y_valid
        # 打印特征的重要性分数
        for i, importance in feature_index_importance:
            # print(f"Feature {i}: {importance}")
            # train_temp_df = train_df[f"feature_{i}"]
            # test_temp_df = test_df[f'feature_{i}']
            # train_test_rate = (train_temp_df.mean() / test_temp_df.mean())
            # if train_test_rate<0.01 or train_test_rate>100:
            print(f"Feature {i}: {importance}")


        # print(self.lgb_clf.booster_.current_iteration())
        # for i in range(3):
        #     dot = lgb.create_tree_digraph(self.lgb_clf, tree_index=i)
        #     dot.view()

        self.predict_eval(X, y, self.lgb_clf)



    def evaluate(self, test_dataset):
        self.logger.info("=========================Start lightGBM evaluating phase ...==============================")

        # load lgb model
        lgb_clf = self.lgb_clf
        X_valid, y_valid, sample_ids = test_dataset
        self.predict_eval(X_valid, y_valid, lgb_clf)
