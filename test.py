import pickle

import numpy as np
import pandas as pd
from joblib._multiprocessing_helpers import mp
from sklearn.metrics import classification_report, f1_score

from core.preprocess.dataset import load_raw_data
from core.preprocess.feature_extractor import *
from main import args
import lightgbm as lgb

from utils.logger import get_logger


# 二分类自定义评估函数
def custom_eval_binary(y_true, y_pred):
    y_pred = [1.0 if y_pred_i>=0.5 else 0.0 for y_pred_i in y_pred]
    return 'f1_score', f1_score(y_true, y_pred, average='macro'), True


# 多分类评估函数
def custom_eval_muti(y_true, y_pred):
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'f1_score', f1_score(y_true, y_pred, average='macro'), True


# 按照类别集合，划分样本 [[0,1,2],[3,4,5]]
def split_class(keeps_list, train_raw_data, test_raw_data):
    train_raw_data_temp = []
    test_raw_data_temp = []
    for id, keeps in enumerate(keeps_list):
        for keep in keeps:
            train_raw_data_temp.append(train_raw_data[lambda df:df['label']==keep])
            train_raw_data_temp[len(train_raw_data_temp)-1]['label']=id
            test_raw_data_temp.append(test_raw_data[lambda df:df['label']==keep])
            test_raw_data_temp[len(test_raw_data_temp)-1]['label']=id
    train_raw_data = pd.concat(train_raw_data_temp)
    test_raw_data = pd.concat(test_raw_data_temp)
    train_dataset = dataset_adapter(train_raw_data,logger)
    test_dataset = dataset_adapter(test_raw_data,logger)
    return train_dataset, test_dataset, train_raw_data


# 训练加评估
def train_evaluate(clf, train_dataset, test_dataset, custom_eval=custom_eval_binary):
    clf.fit(train_dataset[0], train_dataset[1], eval_set=(test_dataset[0], test_dataset[1]), eval_metric=custom_eval)
    pred_labels = clf.predict(train_dataset[0], num_iteration=clf.best_iteration_ + 1)
    logger.debug("\n{}".format(
        classification_report(train_dataset[1], pred_labels, digits=4)
    ))
    # 评估
    pred_labels = clf.predict(test_dataset[0])
    logger.debug("\n{}".format(
        classification_report(test_dataset[1], pred_labels, digits=4)
    ))


# 六种类别预测
def predict_all(clf_list, X):
    pred_labels = []
    for x in X:
        x = np.array([x])
        if clf_list[0].predict(x)[0] == 0:
            if clf_list[1].predict(x)[0] == 0:
                pred_labels.append(0)
            else:
                pred_labels.append(clf_list[2].predict(x)[0]+1)
        else:
            pred_labels.append(clf_list[-1].predict(x)[0]+3)
    return pred_labels


# 六种类别评估
def evaluate_all(clf_list, train_dataset, test_dataset):
    pred_labels = predict_all(clf_list, train_dataset[0])
    logger.debug("\n{}".format(
        classification_report(train_dataset[1], pred_labels, digits=4)
    ))

    pred_labels = predict_all(clf_list, test_dataset[0])
    logger.debug("\n{}".format(
        classification_report(test_dataset[1], pred_labels, digits=4)
    ))


# 保存对象
def dump_pkl(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


# 加载对象
def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# 保存模型
def save_model(clf_list, default_model='./model/MyModel.pkl'):
    dump_pkl(clf_list, default_model)


def get_model(default_model='./model/MyModel.pkl'):
    return load_pkl(default_model)



import numpy as np
from sklearn.manifold import TSNE

# Random state.
RS = 20180101

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(3):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=34)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


"""
X: raw data, 
y: labels
"""

if __name__ != '__main__':
    args.model = 'test'
    logger = get_logger(args.log_dir, args.model)
    train_raw_data, _ = load_raw_data(args)
    train_raw_data.fillna(0, inplace=True)
    X, y,_ = dataset_adapter(train_raw_data, logger)

    digits_proj = TSNE(random_state=RS).fit_transform(X)

    scatter(digits_proj, y)
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('demo.eps', format='eps', dpi=1000)
    plt.show()
    plt.close()

# 只包含1，2两种标签的数据
# 看能不能区分开1和2，即使把差异很大的特征删除了，也不能很好区分1和2
if __name__ == '__main__':
    args.model = 'test'
    logger = get_logger(args.log_dir, args.model)

    # 加载数据
    train_raw_data, test_raw_data = load_raw_data(args)
    train_raw_data = delete_repeated_rows(train_raw_data,logger)

    # 需要训练中模型[1,2,3,4] [0,1,2][3,4,5] [0][1,2] [1][2] [3][4][5]
    needed_train = [4]

    # 训练区分[0,1,2] 和[3,4,5]
    keeps_list_1 = [[0,1,2], [3,4,5]]
    train_dataset_1, test_dataset_1,train_raw_data_1 = split_class(keeps_list_1,train_raw_data,test_raw_data)
    # lightgbm模型

    lgb_params_1 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_iterations': 8000,
        'num_leaves': 511,
        'max_depth': 9,
        'learning_rate': 0.02,
        'n_jobs': mp.cpu_count(),
        'min_data_in_leaf': 50,
        'early_stopping_rounds': 800,
        'metric': ['custom'],
        # 'class_weight': class_weighted(train_raw_data_1),
        'class_weight': 'balanced',
        'reg_lambda': 0.01,
        'feature_fraction': 0.8,
    }
    clf_1 = lgb.LGBMClassifier(**lgb_params_1)
    if 1 in needed_train:
        logger.info("train [[0,1,2], [3,4,5]]")
        # 训练和评估
        train_evaluate(clf_1,train_dataset_1,test_dataset_1)

    # 训练区分[0],[1,2]
    keeps_list_2 = [[0],[1,2]]
    train_dataset_2, test_dataset_2,train_raw_data_2 = split_class(keeps_list_2, train_raw_data, test_raw_data)
    # lightgbm模型

    lgb_params_2 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_iterations': 8000,
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.02,
        'n_jobs': mp.cpu_count(),
        'min_data_in_leaf': 400,
        'early_stopping_rounds': 800,
        'metric': ['custom'],
        # 'class_weight': class_weighted(train_raw_data_2),
        'class_weight': 'balanced',
        'reg_lambda': 0.005,
        'feature_fraction': 0.6,
    }
    clf_2 = lgb.LGBMClassifier(**lgb_params_2)
    if 2 in needed_train:
        logger.info("train [[0],[1,2]]")
        train_evaluate(clf_2, train_dataset_2, test_dataset_2)

    # 训练区分[1],[2]
    keeps_list_3 = [[1], [2]]
    train_dataset_3, test_dataset_3,train_raw_data_3 = split_class(keeps_list_3, train_raw_data, test_raw_data)
    # lightgbm模型

    lgb_params_3 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_iterations': 8000,
        'num_leaves': 127,
        'max_depth': 7,
        'learning_rate': 0.002,
        'n_jobs': mp.cpu_count(),
        'min_data_in_leaf': 20,
        'early_stopping_rounds': 800,
        'metric': ['custom'],
        # 'class_weight': class_weighted(train_raw_data_3),
        'class_weight': 'balanced',
        'reg_lambda': 0.005,
        'feature_fraction': 0.6,
    }
    clf_3 = lgb.LGBMClassifier(**lgb_params_3)
    if 3 in needed_train:
        logger.info("train [[1],[2]]")
        train_evaluate(clf_3, train_dataset_3, test_dataset_3)

    #
    # 训练区分[[3],[4],[5]]
    keeps_list_4 = [[3],[4],[5]]
    train_dataset_4, test_dataset_4, train_raw_data_4 = split_class(keeps_list_4, train_raw_data, test_raw_data)
    # lightgbm模型a

    lgb_params_4 = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_iterations': 8000,
        'num_leaves': 7,
        'max_depth': 3,
        'learning_rate': 0.02,
        'n_jobs': mp.cpu_count(),
        'min_data_in_leaf': 50,
        'early_stopping_rounds': 800,
        'metric': ['custom'],
        # 'class_weight': class_weighted(train_raw_data_4),
        'class_weight': 'balanced',
        'reg_lambda': 0.005,
        # 'feature_fraction': 0.9,
    }
    clf_4 = lgb.LGBMClassifier(**lgb_params_4)
    if 4 in needed_train:
        logger.info("train [[3],[4],[5]]")
        train_evaluate(clf_4,train_dataset_4,test_dataset_4,custom_eval_muti)


    clf_list = [clf_1,clf_2,clf_3,clf_4]

    # 保存模型
    save_model(clf_list)

    # 评估
    if len(needed_train) == 4:
        evaluate_all(clf_list,dataset_adapter(train_raw_data,logger), dataset_adapter(test_raw_data,logger))

        # 测试集
        test_2000 = pd.read_csv("./data/SoftCup/raw/test_2000_x.csv")


        sample_ids = [str(id) for id in test_2000['sample_id'].values]
        # 特征不能是sample_id, label, 以及剔除的特征
        X = test_2000[[col for col in test_2000.columns.tolist() if
                     col != 'sample_id' and col != 'label' ]].values
        y = predict_all(clf_list, X)
        # result = {}
        # for i, label in enumerate(y):
        #     result[sample_ids[i]] = int(label)

        y_static = {i:0 for i in range(6)}
        for i, label in enumerate(y):
            y_static[label] += 1
        print(y_static)



    # 移除一些特征
    # feature_ids = []
    # feature_ids = [104,106,18,63,0,69,30,50,33,43,39,93] #
    # remove_feats=[f'feature{i}' for i in feature_ids]
    # train_raw_data, test_raw_data = remove_adv_feats(train_raw_data,test_raw_data,logger=logger,remove_feats=remove_feats)







    # 获取特征的重要性分数
    # feature_importance = clf.feature_importances_
    # bias = 0
    # feature_index_importance = []
    # for i, importance in enumerate(feature_importance):
    #     if i in feature_ids:
    #         bias += 1
    #     feature_index_importance.append((i+bias, importance))
    #
    # feature_index_importance = sorted(feature_index_importance, key=lambda x: x[1] * -1)
    # train_df = DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[-1])])
    # train_df['label'] = y
    # test_df = DataFrame(X_valid, columns=[f'feature_{i}' for i in range(X.shape[-1])])
    # test_df['label'] = y_valid
    # 打印特征的重要性分数

    # for i, importance in feature_index_importance:
    #     # print(f"Feature {i}: {importance}")
    #     # train_temp_df = train_df[f"feature_{i}"]
    #     # test_temp_df = test_df[f'feature_{i}']
    #     # train_test_rate = (train_temp_df.mean() / test_temp_df.mean())
    #     # if train_test_rate<0.01 or train_test_rate>100:
    #     print(f"Feature {i}: {importance}")
    # print(feature_index_importance)

# 加载模型，输出测试集的统计结果
if __name__ != '__main__':
    clf_list = get_model()
    # 测试集
    test_2000 = pd.read_csv("./data/SoftCup/raw/test_2000_x.csv")

    sample_ids = [str(id) for id in test_2000['sample_id'].values]
    # 特征不能是sample_id, label, 以及剔除的特征
    X = test_2000[[col for col in test_2000.columns.tolist() if
                   col != 'sample_id' and col != 'label']].values
    y = predict_all(clf_list, X)
    # result = {}
    # for i, label in enumerate(y):
    #     result[sample_ids[i]] = int(label)

    y_static = {i: 0 for i in range(6)}
    for i, label in enumerate(y):
        y_static[label] += 1
    print(y_static)
