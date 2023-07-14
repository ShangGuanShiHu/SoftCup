import random
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from numpy import long, NaN

from pandas import DataFrame

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif  # 或其他适合的统计指标
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.utils import shuffle

# 异常值处理，将nan的值转化为0
from utils.adv import get_adv_feats


#
def get_raw_cols(raw_data: DataFrame, is_raw=True):
    raw_cols = raw_data.columns.tolist()
    if is_raw:
        return [col for col in raw_cols if col != 'sample_id' and col != 'label' and '_' not in col]
    else:
        return [col for col in raw_cols if col != 'sample_id' and col != 'label']


def data_fillna(raw_data: DataFrame, logger, fill_all=True, fill_value: Tuple[float, Dict] = 0, fill_dic=None):
    """
    处理dataframe中的nan值
    :param raw_data: 待处理的dataframe
    :param logger: 记录日志的工具
    :param fill_all: 所有列都填充一个值
    :param fill_value: 当所有列都填充一个值时，需要指明这个值，默认为0
    :param fill_dic: 当每个列都有不同添加方式时，需要指明每一列需要添加的值，默认为当前列的平均值
    :return:
    """
    if fill_all:
        logger.debug("abnormal value process: set nan to 0")
        raw_data = raw_data.fillna(fill_value)
        return raw_data
    else:
        logger.debug("abnormal value process: set nan to self-defined")
        if fill_dic is None:
            fill_dic = {}
            raw_data_cols = raw_data.columns
            features = [col for col in raw_data_cols if col != 'sample_id' and col != 'label']
            for feature in features:
                fill_dic[feature] = raw_data[feature].mean()
        raw_data = raw_data.fillna(fill_dic)
        return raw_data, fill_dic


# 异常值处理, 剔除方差为0的列
def delete_abnormal_data(raw_data: DataFrame, logger, del_cols=None):
    raw_data_cols = raw_data.columns
    if del_cols is None:
        del_cols = []
        for col in raw_data_cols:
            if col != 'sample_id' and col != 'label' and raw_data[col].std() == 0:
                del_cols.append(col)
    raw_data = raw_data.drop(columns=del_cols)
    logger.debug(f"abnormal value process: remove the feature list with variance 0: {del_cols}")
    return raw_data, del_cols


# 异常值处理, 剔除包含nan的列
def delete_nan_columns(raw_data: DataFrame, logger, kept_columns=None):
    # raw_data_cols = raw_data.columns
    # if del_cols is None:
    #     del_cols = []
    #     for col in raw_data_cols:
    #         if col != 'sample_id' and col != 'label' and raw_data[col].std() == 0:
    #             del_cols.append(col)
    # raw_data = raw_data.drop(columns=del_cols)
    if kept_columns is None:
        raw_data = raw_data.dropna(axis=1)
        kept_columns = raw_data.columns.tolist()
        logger.debug(f"abnormal value process: feature list without nan: {kept_columns}")
    else:
        logger.debug(f"abnormal value process: feature list without nan: {kept_columns}")
        raw_data = raw_data[kept_columns]
    return raw_data, kept_columns


# 异常值处理，通过3σ将异常值转化为nan
def change_abnormal_value_to_nan(raw_data: DataFrame, logger, sigma_3_param=None):
    logger.debug("abnormal value process use 3σ to change abnormal value to nan")
    if sigma_3_param is None:
        sigma_3_param = {}
        for col in get_raw_cols(raw_data):
            sigma_3_param[col] = {
                'mean': raw_data[col].mean(),
                'std': raw_data[col].std() * 1
            }
            values = [NaN if abs(value - sigma_3_param[col]['mean']) > sigma_3_param[col]['std'] else value for value in
                      raw_data[col].values.tolist()]
            raw_data[col] = values
    else:
        for col in get_raw_cols(raw_data):
            values = [NaN if abs(value - sigma_3_param[col]['mean']) > sigma_3_param[col]['std'] else value for value in
                      raw_data[col].values.tolist()]
            raw_data[col] = values
    return raw_data, sigma_3_param


# 归一化, 对于任意一个值x，归一化结果为 new_x = (x - min(X)) / (max(X) - min(X))
def normalize_data(raw_data: DataFrame, logger, normalize_param=None):
    logger.debug("normalize every feature with max set to 1")
    raw_data_cols = raw_data.columns
    if normalize_param is None:
        normalize_param = {}
        for col in raw_data_cols:
            if col != 'sample_id' and col != 'label':
                normalize_param[col] = {
                    'min': raw_data[col].min(),
                    'max': raw_data[col].max()
                }
                raw_data[col] = (raw_data[col] - raw_data[col].min()) / (
                        raw_data[col].max() - raw_data[col].min())
    else:
        for col in raw_data_cols:
            if col != 'sample_id' and col != 'label':
                raw_data[col] = (raw_data[col] - normalize_param[col]['min']) / (
                        normalize_param[col]['max'] - normalize_param[col]['min'])
    return raw_data, normalize_param


# 标准化, 对于任意一个值x，归一化结果为 new_x = (x - mean(X)) / std(mean)
def standard_data(raw_data: DataFrame, logger, standard_param=None):
    logger.debug("standard every feature")
    raw_data_cols = raw_data.columns
    if standard_param is None:
        standard_param = {}
        for col in raw_data_cols:
            if col != 'sample_id' and col != 'label':
                standard_param[col] = {
                    'mean': raw_data[col].mean(),
                    'std': raw_data[col].std()
                }
                # raw_data[col] = (raw_data[col] - standard_param[col]['mean']) / standard_param[col]['std']
                raw_data[col] = (raw_data[col] - standard_param[col]['mean'])
    else:
        for col in raw_data_cols:
            if col != 'sample_id' and col != 'label':
                # raw_data[col] = (raw_data[col] - standard_param[col]['mean']) / standard_param[col]['std']
                raw_data[col] = (raw_data[col] - standard_param[col]['mean'])
    return raw_data, standard_param

#
def sk_standard_data(raw_data: DataFrame, logger,scaler=None):
    logger.debug("standard every feature")
    features = get_raw_cols(raw_data, is_raw=False)
    X = raw_data[features].values
    y = raw_data['label'].values
    sample_id = raw_data['sample_id'].values
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    raw_data = pd.DataFrame(X, columns=features)
    raw_data['label'] = y
    raw_data['sample_id'] = sample_id
    return raw_data, scaler

# 基于挑选的特征处理数据
def select_feature(raw_data: DataFrame, logger, selected_feature=None):
    if selected_feature is not None:
        regression_cols = ['sample_id']
        regression_cols.extend(selected_feature)
        logger.debug(f'selected top k={len(selected_feature)} relevant features：{regression_cols}')
        regression_cols.append('label')
        raw_data = raw_data[regression_cols]
        return raw_data


# 特征工程
def feature_engineering(raw_data: DataFrame, logger, k: int = 5, inplace=True):
    selector = SelectKBest(score_func=f_classif, k=k)
    raw_data_cols = raw_data.columns
    X = raw_data[[col for col in raw_data_cols if col != 'sample_id' and col != 'label']]
    y = raw_data['label']

    selector.fit_transform(X, y)

    # 特征相关分数
    scores = selector.scores_
    indices = np.argsort(scores)[::-1]
    features = X.columns.tolist()
    features = [(features[indices[i]], scores[indices[i]]) for i in range(len(scores) - 1)]
    logger.debug(f"feature relevant scores: {features}")

    # 获取挑选出的列名
    cols = selector.get_support(indices=True)
    selected_feature = list(X.columns[cols])
    if inplace:
        raw_data = select_feature(raw_data, logger, selected_feature)
    return raw_data, selected_feature


# 数据增强
def data_augment(raw_data: DataFrame, logger, loc=0, std_dev=0.5):
    # logger.debug("data augment by gs noise")
    #
    # mask = np.random.randint(low=0, high=2, size=raw_data.shape)
    # # 别给sample_id和label也打上了噪声
    # if 'label' in raw_data.columns.tolist():
    #     mask[:,-1] = 0
    # if 'sample_id' in raw_data.columns.tolist():
    #     mask[:,0] = 0
    # # mask = np.ones(shape=raw_data.shape)
    # noise = np.random.normal(loc=loc, scale=std_dev, size=raw_data.shape)
    # raw_data = raw_data + mask*noise
    return raw_data


# 随机mask
def random_mask(raw_data: DataFrame, logger, mask_rate_m=2):
    mask = np.random.randint(low=0, high=mask_rate_m, size=raw_data.shape)
    mask[mask != 0] = 1
    if 'label' in raw_data.columns.tolist():
        mask[:, -1] = 1
    if 'sample_id' in raw_data.columns.tolist():
        mask[:, 0] = 1
    raw_data = raw_data * mask
    return raw_data


# 上下采样
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter


def sample_data_adaptor_X(raw_data: DataFrame):
    features = get_raw_cols(raw_data, is_raw=False)
    X = raw_data[features].values
    y = raw_data['label'].values
    return X, y, features


def X_adaptor_sample_data(X_resampled, y_resampled, features):
    raw_data = pd.DataFrame(X_resampled, columns=features)
    raw_data['label'] = y_resampled
    raw_data['sample_id'] = NaN
    return raw_data


def random_over_sample(logger, raw_data: DataFrame):
    X, y, features = sample_data_adaptor_X(raw_data)
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    logger.debug(sorted(Counter(y_resampled).items()))

    return X_adaptor_sample_data(X_resampled,y_resampled,features)


def random_under_sample(logger, raw_data: DataFrame):
    X, y, features = sample_data_adaptor_X(raw_data)
    ros = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    logger.debug(sorted(Counter(y_resampled).items()))

    return X_adaptor_sample_data(X_resampled, y_resampled, features)


def smote_over_sample(logger, raw_data: DataFrame):
    X, y, features = sample_data_adaptor_X(raw_data)
    X_resampled_smote, y_resampled_smote = SMOTE().fit_resample(X, y)
    logger.debug(sorted(Counter(y_resampled_smote).items()))

    return X_adaptor_sample_data(X_resampled_smote, y_resampled_smote, features)


from imblearn.over_sampling import BorderlineSMOTE
def borderlinesmote_over_sample(logger, raw_data: DataFrame):
    X, y, features = sample_data_adaptor_X(raw_data)
    bsmote = BorderlineSMOTE()
    X_resampled, y_resampled = bsmote.fit_resample(X, y)
    logger.debug(sorted(Counter(y_resampled).items()))

    return X_adaptor_sample_data(X_resampled, y_resampled, features)


def adysn_over_sample(logger, raw_data: DataFrame):
    X, y, features = sample_data_adaptor_X(raw_data)
    X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    logger.debug(sorted(Counter(y_resampled).items()))

    return X_adaptor_sample_data(X_resampled, y_resampled, features)


def combination(logger, raw_data: DataFrame):
    X, y, features = sample_data_adaptor_X(raw_data)
    over = RandomOverSampler()
    under = RandomUnderSampler()
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    logger.debug(sorted(Counter(y_resampled).items()))

    return X_adaptor_sample_data(X_resampled, y_resampled, features)


# 均衡样本数量
def balance_class_number(raw_datas: DataFrame, logger, c_num=500, ignore=None, keep=None, sample_num=None):
    logger.debug(f"balance class number: set every kind num to {c_num}")
    label_data_first = None
    for label, label_data in raw_datas.groupby('label'):
        if sample_num is not None:
            c_num = sample_num[label]
        if keep is not None and label not in keep:
            continue
        if ignore is None or label not in ignore:
            if len(label_data) < c_num:
                label_data = pd.concat(
                    [label_data, data_augment(label_data.sample(c_num - len(label_data), replace=True), logger)])
            else:
                label_data = label_data.sample(n=c_num)
        if label_data_first is None:
            label_data_first = label_data
        else:
            label_data_first = label_data_first.append(label_data)
    return label_data_first


# 统计每行的nan值
def statistic_nan_num(raw_datas: DataFrame, logger):
    logger.debug(f"statistic nan count in every sample")
    nan_nums = []
    for idx, raw_data in raw_datas.iterrows():
        nan_num = 0
        for isnan in raw_data.isna():
            if isnan:
                nan_num += 1
        nan_nums.append(nan_num)
    raw_datas['feature107'] = nan_nums
    return raw_datas


# 对抗验证，去除一些数据，让训练集和测试并不是那么容易被分开
def remove_adv_feats(train_raw_datas: DataFrame, test_raw_datas: DataFrame, logger, remove_feats=None):
    if remove_feats is None:
        raw_data_cols = train_raw_datas.columns
        feats = [col for col in raw_data_cols if col != 'sample_id' and col != 'label' and "_" not in col]
        new_feats, remove_feats = get_adv_feats(train_raw_datas, test_raw_datas, feats)
        logger.debug(f"feature engineering: remove adv feats {remove_feats}")
    return train_raw_datas.drop(columns=remove_feats), test_raw_datas.drop(columns=remove_feats)


# 用高斯噪声来替代空值
def fill_nan_with_noisy(raw_datas: DataFrame, logger, loc=0, std_dev=0.1):
    logger.debug("replace nan with gs noise")
    train_nan_place = (raw_datas - raw_datas).fillna(1)
    noise = train_nan_place * np.random.normal(loc=loc, scale=std_dev, size=raw_datas.shape)
    # 添加高斯噪声
    raw_datas = data_fillna(raw_datas, logger)
    raw_datas = raw_datas + noise
    return raw_datas


# 观察数据发现，有很多特征，大部分值是保持同一个数，有理由认为如果不是这个数，就是异常，以此作为衍生特征
def abnormal_value_not_frequent(raw_datas: DataFrame, logger, frequent_dic=None, frequent_rate=0.8):
    features: List[str] = get_raw_cols(raw_datas)
    if frequent_dic is None:
        frequent_dic = {}
        for feature in features:
            new_feature = f"{feature}_nf"
            key, num = list(raw_datas[feature].value_counts().to_dict().items())[0]
            data_length = int(raw_datas[feature].count() * frequent_rate)
            if num >= data_length:
                frequent_dic[feature] = key
                new_feature_value = []
                for feature_value in raw_datas[feature]:
                    if feature_value == frequent_dic[feature]:
                        new_feature_value.append(0)
                    elif np.isnan(feature_value):
                        new_feature_value.append(1)
                    else:
                        new_feature_value.append(1)
                raw_datas[new_feature] = new_feature_value
    else:
        for feature in features:
            if feature in frequent_dic:
                new_feature = f"{feature}_nf"
                new_feature_value = []
                for feature_value in raw_datas[feature]:
                    if feature_value == frequent_dic[feature]:
                        new_feature_value.append(0)
                    elif np.isnan(feature_value):
                        new_feature_value.append(1)
                    else:
                        new_feature_value.append(1)
                raw_datas[new_feature] = new_feature_value
    logger.debug("abnormal process: frequent value")
    return raw_datas, frequent_dic


# 异常值处理，对正常值、异常值以及空值进行三分类，异常值处理通过3σ来得到，具体多少σ未知
def abnormal_value_one_hot(raw_data: DataFrame, logger, sigma_param=None, sigma_num=2, ignore_feature=None):
    logger.debug("abnormal value process use 3σ to classify abnormal value, normal value and nan")
    features: List[str] = get_raw_cols(raw_data)
    if sigma_param is None:
        sigma_param = {}
        for col in features:
            new_feature = f"{col}_ab"
            sigma_param[col] = {
                'mean': raw_data[col].mean(),
                'std': raw_data[col].std() * sigma_num
            }
            values = []
            for value in raw_data[col]:
                if abs(value - sigma_param[col]['mean']) < sigma_param[col]['std']:
                    values.append(0)
                elif np.isnan(value):
                    values.append(NaN)
                else:
                    values.append(1)
            raw_data[new_feature] = values
    else:
        for col in features:
            new_feature = f"{col}_ab"
            values = []
            for value in raw_data[col]:
                if abs(value - sigma_param[col]['mean']) < sigma_param[col]['std']:
                    values.append(0)
                elif np.isnan(value):
                    values.append(NaN)
                else:
                    values.append(1)
            raw_data[new_feature] = values
    return raw_data, sigma_param


# 多项式得组合特征
def combination_features(raw_data: DataFrame, logger):
    logger.debug("feature engineering: combination features")
    # 创建多项式特征组合
    poly = PolynomialFeatures(degree=2, include_bias=False)
    raw_features = get_raw_cols(raw_data)
    X_df = raw_data[raw_features]
    transformed_data = poly.fit_transform(X_df)
    new_features = poly.get_feature_names_out(raw_features)
    transformed_df = pd.DataFrame(transformed_data, columns=new_features)
    raw_data = raw_data.drop(columns=raw_features)
    raw_data[new_features] = transformed_df
    return raw_data

# 模型数据
def dataset_adapter(raw_data: DataFrame, logger):
    raw_data = shuffle(raw_data)
    X = np.array(raw_data[get_raw_cols(raw_data, False)].values)
    y = np.array(raw_data['label'].values, dtype=long)
    sample_id = np.array(raw_data['sample_id'].values, dtype=long)
    return tuple([X, y, sample_id])

# 数据偏移
def remove_features_by_train_test(train_raw_datas: DataFrame, test_raw_datas: DataFrame, logger, feats = None):
    if feats is None:
        feats = remove_adv_feats(train_raw_datas,test_raw_datas,logger)
    for feat in feats:
        # train_mean = train_raw_datas[feat].mean()
        # train_std = train_raw_datas[feat].std()
        # test_mean = test_raw_datas[feat].mean()
        # test_std = test_raw_datas[feat].std()
        #
        # # 分布平移
        # # test_raw_datas[feat] = ((test_raw_datas[feat].values - test_mean)/test_std)*train_std + train_mean
        # test_raw_datas[feat] = test_raw_datas[feat].values - test_mean + train_mean
        train_p50 = train_raw_datas[feat].quantile(0.5)
        test_p50 = test_raw_datas[feat].quantile(0.5)
        test_raw_datas[feat] = test_raw_datas[feat].values - test_p50 + train_p50
    return train_raw_datas, test_raw_datas


