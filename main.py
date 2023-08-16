import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import warnings

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from core.approach.catboost import CatBoost
from core.approach.fc import FC
from core.approach.lightgbm import LightGBM,custom_eval
from core.approach.randomforest import RandomForest
from core.approach.softcup_dataset import SoftCupDataset
from core.approach.svm import SVM
from core.approach.mlp import MLP
from core.preprocess.dataset import load_raw_data
from core.preprocess.feature_extractor import *
from utils.logger import get_logger

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# common args
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu_devices', type=str, default='0')
# 定义训练还是验证'train','vaild'
parser.add_argument('--task_type', type=str, default='vaild')

# model有lightgbm、svm、mlp、randomforest,fc,catboost
parser.add_argument('--model', type=str, default='lightgbm')
parser.add_argument('--log_dir', type=str, default='./output')
parser.add_argument('--model_dir', type=str, default='./model')

# dataset
parser.add_argument('--dataset', type=str, default='./SoftCup')
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--batch_size', type=int, default=32)

# data process
parser.add_argument('--K', type=int, default=40)
parser.add_argument('--c_num', type=int, default=490)

# lgb
parser.add_argument('--objective', type=str, default='multiclass')
# parser.add_argument('--objective', type=str, default='binary')
parser.add_argument('--boosting_type', type=str, default='gbdt')
parser.add_argument('--lgb_learning_rate', type=float, default=0.01)
parser.add_argument('--num_iterations', type=int, default=8000)
# 叶子节点，不等同于分类类别，等同于分叉的次数
parser.add_argument('--num_leaves', type=int, default=6)
parser.add_argument('--min_data_in_leaf', type=int,default=30)
parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--early_stopping_rounds', type=int, default=1000)
parser.add_argument('--valid', type=bool, default=False)


# SVM
# 对于一些分类错误样本的惩罚
parser.add_argument('--C', type=float, default=10)
# 核函数,'rbf',linear,poly
parser.add_argument('--kernel', type=str, default="rbf")
parser.add_argument('--decision_function_shape', type=str, default='ovr')

# MLP
parser.add_argument('--hidden_layer_sizes', type=int, default=100)
parser.add_argument('--max_iter', type=int, default=2000)
parser.add_argument('--random_state', type=int, default=42)

# fc
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--out_dim', type=int, default=6)
parser.add_argument('--in_dim', type=int, default=107)
parser.add_argument('--hidden', type=list, default=[64,32])
parser.add_argument('--dropout', type=float, default=0.25)

args = parser.parse_args()

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 网格调参
def parameter_experiment(logger):
    logger.info("==============Load dataset {}===============".format(args.dataset))
    train_raw_data, test_raw_data = load_raw_data(args)

    logger.info("==============Data process===============")
    train_raw_data = balance_class_number(train_raw_data,logger=logger,c_num=args.c_num)

    train_dateset = dataset_adapter(train_raw_data,logger)
    test_dateset = dataset_adapter(test_raw_data,logger)


    logger.info("==============init model================")
    clf = LightGBM(args).lgb_clf
    # 定义参数网格
    param_grid = {
        'num_leaves': [6]
        # 'num_leaves': [4, 6, 8, 10]
        # 'learning_rate': [0.005, 0.01, 0.015],
        # 'num_iterations': [2000, 3000, 4000],
        # 'max_depth': [3,4,5,6],
        # 'min_data_in_leaf': [20,30,40,50]
    }
    from sklearn.model_selection import GridSearchCV
    clf_gscv = GridSearchCV(clf, param_grid, cv=5)

    X, y, sample_ids = train_dateset


    X_valid, y_valid, sample_ids = test_dateset

    clf_gscv.fit(X, y, eval_set=[(X_valid, y_valid)], eval_metric=custom_eval)

    pred_probs = clf_gscv.predict(X_valid)
    # pred_labels = np.argmax(pred_probs, axis=1)
    pred_labels = pred_probs

    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(y_valid, pred_labels, average='macro')
    pre = precision_score(y_valid, pred_labels, average='macro')
    rec = recall_score(y_valid, pred_labels, average='macro')

    print("Best parameters: ", clf_gscv.best_params_)
    print("Best score: ", clf_gscv.best_score_)
    logger.debug("[Failure Type Identification]")
    logger.debug("F1, Precison, Recall: {:.3%} {:.3%} {:.3%}".format(f1, pre, rec))
    logger.debug("Best parameters: {}".format(clf_gscv.best_params_))
    logger.debug("Best score: {}".format(clf_gscv.best_score_))

# 数据处理方法实验
def main():
    # logger = get_logger(args.log_dir, 'experiment')
    # 日志
    logger = get_logger(args.log_dir, args.model)
    use_gpu = torch.cuda.is_available()
    logger.info(
        "===========================================================environment configuration=====================================================================================")
    if use_gpu:
        logger.info("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
    else:
        logger.info("Currently using CPU (GPU is highly recommended)")
    # 设置随机种子，保证结果的可复现
    set_seed(args.seed)
    logger.info("======================Load dataset {}===============".format(args.dataset))
    # 加载数据
    train_raw_data, test_raw_data = load_raw_data(args)

    logger.info("==============Data process===============")

    # 将空值转化为0，这是不可取的
    # train_raw_data = data_fillna(train_raw_data, logger)
    # test_raw_data = data_fillna(test_raw_data, logger)

    # 将包含空值的列剔除
    # train_raw_data, kept_columns = delete_nan_columns(train_raw_data, logger)
    # test_raw_data, _ = delete_nan_columns(test_raw_data, logger, kept_columns)

    # 异常值处理，通过3σ将异常值转化为nan
    # train_raw_data, sigma_3_param = change_abnormal_value_to_nan(train_raw_data,logger)
    # test_raw_data,_ = change_abnormal_value_to_nan(test_raw_data,logger, sigma_3_param)

    # 将方差为0的列移除
    train_raw_data, del_cols = delete_abnormal_data(train_raw_data, logger)
    test_raw_data,_ = delete_abnormal_data(test_raw_data, logger, del_cols)

    # 移除训练集中重复的行
    train_raw_data = delete_repeated_rows(train_raw_data,logger)

    # 归一化每一列的数据
    # train_raw_data, normalize_param = normalize_data(train_raw_data, logger)
    # # test_raw_data,_ = normalize_data(test_raw_data, logger, normalize_param)
    # test_raw_data, _ = normalize_data(test_raw_data, logger)

    # 标准化每一列数据
    # train_raw_data, standard_param = standard_data(train_raw_data, logger)
    # test_raw_data, _ = standard_data(test_raw_data, logger)
    # test_raw_data,_ = standard_data(test_raw_data, logger, standard_param)

    # 将每一列的空值转化为该列的平均值, 归一化之前或者之后都可以
    # train_raw_data, fill_dic = data_fillna(train_raw_data, logger, fill_all=False)
    # test_raw_data, fill_dic = data_fillna(test_raw_data, logger, fill_all=False,fill_dic=fill_dic)

    # 特征工程挑选topk相关的列
    # k = args.K
    # train_raw_data, selected_feature = feature_engineering(train_raw_data, logger, k)
    # # test_raw_data = select_feature(test_raw_data, logger, selected_feature)
    # test_raw_data, selected_feature_ = feature_engineering(test_raw_data, logger, k)

    # print(list((set(selected_feature))&(set(selected_feature_))))
    # selected_feature = ['feature22', 'feature45', 'feature14', 'feature10', 'feature36', 'feature31', 'feature16', 'feature33', 'feature71', 'feature105', 'feature30', 'feature59', 'feature46', 'feature27', 'feature13', 'feature97', 'feature15', 'feature67', 'feature61', 'feature19', 'feature87', 'feature44', 'feature35', 'feature5', 'feature49', 'feature85', 'feature2', 'feature76']
    # train_raw_data = select_feature(train_raw_data, logger, selected_feature)
    # test_raw_data = select_feature(test_raw_data, logger, selected_feature)

    # 统计每一行nan值
    # train_raw_data = statistic_nan_num(train_raw_data, logger)
    # test_raw_data = statistic_nan_num(test_raw_data, logger)

    # 均匀样本数量
    # keep_list = [1,2,3]
    # train_raw_data = balance_class_number(train_raw_data, logger, args.c_num, keep=keep_list,ignore=keep_list)
    # test_raw_data = balance_class_number(test_raw_data, logger, args.c_num, keep=keep_list,ignore=keep_list)
    # train_raw_data = balance_class_number(train_raw_data, logger, args.c_num)

    # train_raw_data, frequent_dic = abnormal_value_not_frequent(train_raw_data,logger)
    # test_raw_data, _ = abnormal_value_not_frequent(test_raw_data,logger)
    # train_raw_data, sigma_param = abnormal_value_one_hot(train_raw_data, logger)
    # test_raw_data, _ = abnormal_value_one_hot(test_raw_data,logger)

    # 下采样
    # train_raw_data = balance_class_number(train_raw_data, logger, args.c_num, ignore=[1,2], keep=[0,1,2])
    # test_raw_data = balance_class_number(test_raw_data,logger,args.c_num, ignore=[0,1,2], keep=[0,1,2])

    # removed_feats_valid = [f'feature{i}' for i in
    #                  [0, 11, 17, 18, 20, 24, 26, 28, 32, 38, 40, 50, 54, 60, 62, 63, 64, 65, 69, 70, 73, 74, 78, 80, 88,
    #                   90, 93, 98, 99, 103, 104, 106]]
    # train_raw_data, test_raw_data = remove_features_by_train_test(train_raw_data,test_raw_data,logger,removed_feats_valid)
    # train_raw_data = balance_class_number(train_raw_data,logger,args.c_num)



    # 移除一些容易区分训练集和测试集的特征
    # train_raw_data, test_raw_data = remove_adv_feats(train_raw_data, test_raw_data, logger)
    # removed_feats = [f'feature{i}' for i in [0,2,11,17,18,20,24,26,28,32,38,40,50,54,60,62,63,64,65,69,70,73,74,78,80,88,90,92,93,98,99,103,104,106]]
    # removed_feats = [f'feature{i}' for i in
    #                  [0, 2, 11, 17, 18, 20, 24, 26, 28, 32, 38, 40, 50, 54, 60, 62, 63, 64, 65, 69, 70, 73, 74, 78, 80,
    #                   88, 90, 92, 93, 98, 99, 103, 104, 106]]
    # removed_feats_valid = [f'feature{i}' for i in
    #                  [0, 11, 17, 18, 20, 24, 26, 28, 32, 38, 40, 50, 54, 60, 62, 63, 64, 65, 69, 70, 73, 74, 78, 80, 88,
    #                   90, 93, 98, 99, 103, 104, 106]]
    # removed_feats_test = ['feature3', 'feature12', 'feature18', 'feature23', 'feature24', 'feature25', 'feature28', 'feature29', 'feature34', 'feature37', 'feature40', 'feature41', 'feature47', 'feature50', 'feature53', 'feature62', 'feature63', 'feature68', 'feature69', 'feature70', 'feature74', 'feature83', 'feature90', 'feature93', 'feature95', 'feature99', 'feature102', 'feature103']
    # removed_feats = sorted(list(set(removed_feats_valid).union(set(removed_feats_test))))
    #
    # train_raw_data, test_raw_data = remove_adv_feats(train_raw_data, test_raw_data, logger, removed_feats)


    # 下采样
    # train_raw_data = random_under_sample(logger, train_raw_data)

    # 衍生特征，区分频繁出现得值和非频繁出现的值，进one-hot编码
    # train_raw_data, frequent_dic = abnormal_value_not_frequent(train_raw_data,logger)
    # test_raw_data, _ = abnormal_value_not_frequent(test_raw_data,logger)

    # 衍生特征，异常值处理，对正常值、异常值以及空值进行三分类，异常值处理通过3σ来得到，具体多少σ未知
    # train_raw_data, sigma_param = abnormal_value_one_hot(train_raw_data, logger)
    # test_raw_data, _ = abnormal_value_one_hot(test_raw_data,logger)
    # train_raw_data,test_raw_data, imp = data_fillna_mean(train_raw_data,test_raw_data)

    # train_raw_data,test_raw_data, pca = extract_feature_pca(train_df=train_raw_data,test_df=test_raw_data,logger=logger,n_features=30)
    # 上采样
    # train_raw_data = random_over_sample(logger, train_raw_data)

    # 联合上下采样
    # train_raw_data = combination(logger, train_raw_data)

    # train_raw_data = borderlinesmote_over_sample(logger, train_raw_data)

    # train_raw_data = random_over_sample(logger, train_raw_data)
    # train_raw_data = random_under_sample(logger, train_raw_data)

    # train_raw_data = combination_features(train_raw_data, logger)
    # test_raw_data = combination_features(test_raw_data, logger)


    # 将dataframe格式的总数据转化为模型能够训练的数据
    train_dataset = dataset_adapter(train_raw_data, logger)
    test_dataset = dataset_adapter(test_raw_data, logger)

    logger.info(f"==============init {args.model} model================")
    if args.model == 'lightgbm':
        clf = LightGBM(args)
    elif args.model == "svm":
        clf = SVM(args)
    elif args.model == "mlp":
        clf = MLP(args)
    elif args.model == 'randomforest':
        clf = RandomForest(args)
    elif args.model == 'catboost':
        clf = CatBoost(args)
    else:
        raise NotImplementedError
    clf.fit(train_dataset, test_dataset)
    clf.evaluate(test_dataset)


# 这里是对空值进行了处理，神经网络
def train_fc():
    logger = get_logger(args.log_dir, args.model)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        logger.info("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
    else:
        logger.info("Currently using CPU (GPU is highly recommended)")
    set_seed(args.seed)

    train_raw_data, test_raw_data = load_raw_data(args)

    # 将空值转化为0，这是不可取的
    # train_raw_data = data_fillna(train_raw_data, logger)
    # test_raw_data = data_fillna(test_raw_data, logger)

    # 将包含空值的列剔除
    # train_raw_data, kept_columns = delete_nan_columns(train_raw_data, logger)
    # test_raw_data, _ = delete_nan_columns(test_raw_data, logger, kept_columns)

    # 将方差为0的列移除
    # train_raw_data, del_cols = delete_abnormal_data(train_raw_data, logger)
    # test_raw_data,_ = delete_abnormal_data(test_raw_data, logger, del_cols)

    # 归一化每一列的数据
    # train_raw_data, normalize_param = normalize_data(train_raw_data, logger)
    # test_raw_data,_ = normalize_data(test_raw_data, logger, normalize_param)
    # test_raw_data, _ = normalize_data(test_raw_data, logger)

    # 标准化每一列数据
    # train_raw_data, standard_param = standard_data(train_raw_data, logger)
    # test_raw_data,_ = standard_data(test_raw_data,logger)
    # test_raw_data,_ = standard_data(test_raw_data, logger, standard_param)

    # _, scalar = sk_standard_data(pd.concat([train_raw_data, test_raw_data]), logger)
    # train_raw_data,scaler = sk_standard_data(train_raw_data,logger, scalar)

    # removed_feats = [f'feature{i}' for i in [0,11,17,18,20,24,26,28,32,38,40,50,54,60,62,63,64,65,69,70,73,74,78,80,88,90,93,98,99,103,104,106]]
    # train_raw_data, test_raw_data = remove_adv_feats(train_raw_data, test_raw_data, logger,removed_feats)

    # train_raw_data, scaler = sk_standard_data(train_raw_data, logger)
    # test_raw_data,_ = sk_standard_data(test_raw_data,logger)

    # 将空值转化为0，这是不可取的
    train_raw_data = data_fillna(train_raw_data, logger)
    test_raw_data = data_fillna(test_raw_data, logger)
    # test_raw_data = data_fillna(test_raw_data, logger)

    # train_raw_data, frequent_dic = abnormal_value_not_frequent(train_raw_data,logger)
    # test_raw_data, _ = abnormal_value_not_frequent(test_raw_data,logger)

    # train_raw_data, sigma_param = abnormal_value_one_hot(train_raw_data, logger)
    # test_raw_data, _ = abnormal_value_one_hot(test_raw_data,logger)

    # 添加高斯噪声的标准差（控制噪声强度）
    # 将nan值用高斯噪声代替
    # train_raw_data = fill_nan_with_noisy(train_raw_data, logger)
    # test_raw_data = fill_nan_with_noisy(test_raw_data,logger)

    # 将每一列的空值转化为该列的平均值, 归一化之前或者之后都可以
    # train_raw_data, fill_dic = data_fillna(train_raw_data, logger, fill_all=False)
    # test_raw_data, fill_dic = data_fillna(test_raw_data, logger, fill_all=False, fill_dic=fill_dic)
    # test_raw_data, fill_dic = data_fillna(test_raw_data, logger, fill_all=False)

    # 特征工程挑选topk相关的列
    # k = args.in_dim
    # train_raw_data, selected_feature = feature_engineering(train_raw_data, logger, k)
    # test_raw_data = select_feature(test_raw_data, logger, selected_feature)
    # train_raw_data, sigma_3_param = change_abnormal_value_to_nan(train_raw_data,logger)
    # test_raw_data,_ = change_abnormal_value_to_nan(test_raw_data,logger, sigma_3_param)

    # 均匀样本数量
    # sample_num = {
    #     0.0: 5144, # 5144
    #     1.0: 1613, # 1613
    #     2.0: 1062, # 1062
    #     3.0: 884, # 884
    #     4.0: 554, # 554
    #     5.0: 743 # 743
    # }
    # train_raw_data = balance_class_number(train_raw_data, logger, args.c_num, sample_num=sample_num)
    # keep_list = [1,2]
    # train_raw_data = balance_class_number(train_raw_data, logger, args.c_num, ignore=keep_list, keep=keep_list)
    # test_raw_data = balance_class_number(test_raw_data,logger,args.c_num, ignore=keep_list, keep=keep_list)



    #
    # keeps_list = [[1],[2]]

    # keeps_list = [[i] for i in range(6)]
    # train_raw_data_temp = []
    # test_raw_data_temp = []
    # for id, keeps in enumerate(keeps_list):
    #     for keep in keeps:
    #         train_raw_data_temp.append(train_raw_data[lambda df:df['label']==keep])
    #         train_raw_data_temp[len(train_raw_data_temp)-1]['label']=id
    #         test_raw_data_temp.append(test_raw_data[lambda df:df['label']==keep])
    #         test_raw_data_temp[len(test_raw_data_temp)-1]['label']=id
    # train_raw_data = pd.concat(train_raw_data_temp)
    # test_raw_data = pd.concat(test_raw_data_temp)

    # train_raw_data = data_augment(train_raw_data,logger)
    # test_raw_data = data_augment(test_raw_data,logger)

    # 随机mask空值
    # train_raw_data = pd.concat([train_raw_data, random_mask(train_raw_data, logger, mask_rate_m=2)])

    # train_dataset_ = SoftCupDataset(train_raw_data)
    # dataset_size = len(train_dataset_)
    # indices = list(range(dataset_size))
    # split = int(np.floor(args.train_ratio * dataset_size))
    # np.random.shuffle(indices)
    #
    # train_indices, val_indices = indices[:split], indices[split:]
    #
    # train_dataset = Subset(train_dataset_, train_indices)
    # test_dataset = Subset(train_dataset_, val_indices)

    logger.info(f"==============init {args.model} model================")
    if args.model=='fc':
        train_dataset = SoftCupDataset(train_raw_data)
        test_dataset = SoftCupDataset(test_raw_data)
        #
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        fc_clf = FC(args)
        fc_clf.fit(train_loader,test_loader)
        # fc_clf = FC(args)
        # fc_clf.fit_k(train_raw_data,test_raw_data)
        # fc_clf.evaluate(test_loader)
        # fc_clf.evaluate(test_loader)

    else:
        if args.model == "svm":
            clf = SVM(args)
        elif args.model == 'randomforest':
            clf = RandomForest(args)
        elif args.model == "mlp":
            clf = MLP(args)
        else:
            raise NotImplementedError
        # 将dataframe格式的总数据转化为模型能够训练的数据
        train_dataset = dataset_adapter(train_raw_data, logger)
        test_dataset = dataset_adapter(test_raw_data, logger)
        clf.fit(train_dataset,test_dataset)
        clf.evaluate(test_dataset)

# 查看对抗验证的效果
def adv_feat_experiment():
    from utils.adv import get_adv_feats
    logger = get_logger(args.log_dir, 'feat-experiment')
    train_raw_data, test_raw_data = load_raw_data(args)
    train_raw_data = balance_class_number(train_raw_data, logger, args.c_num)

    raw_data_cols = train_raw_data.columns
    feats = [col for col in raw_data_cols if col != 'sample_id' and col != 'label']
    new_feats,remove_feats = get_adv_feats(train_raw_data,test_raw_data,feats)
    print("=========================================================")
    print(new_feats)


if __name__ == "__main__":
    main()
    # train_fc()
    # logger = get_logger(args.log_dir, 'parameter_experiment')
    # parameter_experiment(logger)
    # adv_feat_experiment()


# kde技术
# import seaborn as sns
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     train_raw_data, test_raw_data = load_raw_data(args)
#     # removed_feats_valid = ['feature3', 'feature4','feature5','feature6','feature7','feature8','feature9','feature10']
#     # removed_feats_valid = []
#     # removed_feats_valid.extend([f'feature{i}' for i in
#     #                  [0, 11, 17, 18, 20, 24, 26, 28, 32, 38, 40, 50, 54, 60, 62, 63, 64, 65, 69, 70, 73, 74, 78, 80, 88,
#     #                   90, 93, 98, 99, 103, 104, 106]])
#     removed_feats_valid = [(2, 1492), (45, 1488), (10, 1479), (7, 1160), (104, 1112), (106, 931), (37, 930), (85, 914),
#                            (87, 912), (9, 861), (52, 845), (66, 842), (14, 837), (71, 831), (41, 825), (18, 774),
#                            (95, 757), (63, 695), (89, 694), (96, 690), (29, 678), (72, 676), (12, 657), (81, 650),
#                            (79, 649), (15, 644), (0, 643), (53, 642), (90, 619), (69, 617), (84, 607), (47, 602),
#                            (30, 572), (103, 566), (51, 550), (50, 547), (93, 539), (62, 538), (86, 528), (21, 526),
#                            (101, 522), (17, 518), (67, 515), (49, 514), (99, 514), (44, 508), (38, 499), (83, 486),
#                            (24, 484), (105, 483), (102, 469), (19, 455), (48, 454), (98, 452), (73, 449), (11, 448),
#                            (33, 432), (43, 431), (27, 429), (58, 426), (82, 422), (59, 406), (68, 406), (28, 394),
#                            (39, 389), (8, 373), (35, 371), (40, 339), (4, 337), (25, 337), (3, 336), (97, 332),
#                            (23, 324), (42, 306), (46, 306), (70, 303), (31, 301), (94, 299), (91, 269), (75, 260),
#                            (5, 259), (61, 248), (36, 244), (26, 237), (76, 209), (56, 204), (22, 183), (74, 183),
#                            (34, 151), (6, 150), (55, 150), (13, 131), (16, 87), (60, 51), (20, 21), (1, 20), (80, 17),
#                            (64, 15), (54, 10), (32, 5), (92, 1), (57, 0), (65, 0), (77, 0), (78, 0), (88, 0), (100, 0)]
#
#
#     # feats = [feat for feat in train_raw_data.columns.tolist() if feat not in removed_feats_valid]
#     # for feat in feats:
#     for feat, importance in removed_feats_valid:
#         feat = f'feature{feat}'
#     # feat = 'label'
#         train_feat = train_raw_data[feat].values.tolist()
#         test_feat = test_raw_data[feat].values.tolist()
#         sns.kdeplot(train_feat, shade=True, color='r', label='train')
#         sns.kdeplot(test_feat, shade=True, color='b', label='test')
#         plt.xlabel(f'{feat}-{importance}')
#         plt.legend()
#         plt.title(feat)
#         plt.show()
#         plt.close()

# if __name__ == "__main__":
#
#
#     fc_clf = FC(args)
#     fc_clf.load_model(fc_clf.model_save_dir)
#     fc_clf.model.eval()
#
#     logger = get_logger(args.log_dir, 'fc')
#
#     test_df = pd.read_csv("./data/SoftCup/raw/test.csv")
#
#     # 去掉方差为0的列
#     del_cols = ['feature57', 'feature77', 'feature100']
#     test_df, _ = delete_abnormal_data(test_df, logger, del_cols)
#
#     # 标准化
#     test_df, _ = sk_standard_data(test_df, logger)
#
#     # 将空值转化为0，这是不可取的
#     test_df = data_fillna(test_df, logger)
#
#     # 衍生特征，异常值归类，非异常值归类
#     test_df, _ = abnormal_value_one_hot(test_df, logger)
#
#
#     data = test_df[[col for col in test_df.columns if col != 'sample_id' and col != 'label']].values
#     sample_ids = [int(sample_id) for sample_id in test_df['sample_id'].values]
#     model_logits = fc_clf.model(torch.tensor(data))
#     # pred_labels = np.argmax(model_logits.detach().cpu().numpy(),axis=1)
#     pred_labels = np.argmax(fc_clf.get_cls(model_logits.detach()).cpu().numpy(), axis=1)
#     y_preds = [int(pred_label) for pred_label in pred_labels]
#
#     res = {sample_id: y_pred for (sample_id, y_pred) in zip(sample_ids, y_preds)}
#     print(res)



# from fastapi import FastAPI, UploadFile
# import uvicorn as uvicorn
# import pandas as pd
# from pandas._typing import ReadCsvBuffer
# import lightgbm as lgb
# import numpy as np
#
# app = FastAPI()
#
# def predict(model, data_df):
#     sample_ids = [str(id) for id in data_df['sample_id'].values]
#     X = data_df[[col for col in data_df.columns.tolist() if col!='sample_id' and col!='label' and col not in ['feature8', 'feature13', 'feature9', 'feature33', 'feature60', 'feature89']]].values
#
#     y = model.predict(X)
#     y = np.argmax(y,axis=1)
#     result = {}
#     for i, label in enumerate(y):
#         result[sample_ids[i]] = int(label)
#     return result
#
# def get_model(default_model='./model/lgb'):
#     lgb_clf = lgb.Booster(
#         model_file=default_model
#     )
#     return lgb_clf
#
#
# @app.get("/")
# async def root():
#     return {"message": "Hello, FastAPI!"}
#
#
# @app.post("/upload")
# async def upload_file(file: UploadFile = UploadFile(...)):
#     content = await file.read()  # Read the file content
#
#     # Convert the string content to DataFrame using pandas
#     from io import BytesIO
#     df = pd.read_csv(BytesIO(content))
#     model = get_model()
#     result = predict(model,df)
#     return result
#
#
# @app.post("/train")
# async def train(file: UploadFile = UploadFile(...)):
#     content = await file.read()  # Read the file content
#
#     # Convert the string content to DataFrame using pandas
#     from io import BytesIO
#     df = pd.read_csv(BytesIO(content))
#
# if __name__ == "__main__":
#     uvicorn.run(app, port=8888)
