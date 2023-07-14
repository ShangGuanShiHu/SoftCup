import pandas as pd
import multiprocessing as mp
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def split_dataset(df_train, df_test, train_p):
    # 对抗验证
    df_train['target'] = 0
    df_test['target'] = 1

    df = pd.concat(( df_train, df_test ))

    X = df.drop( [ 'target','label' ], axis = 1 ).values
    y = df.target.values

    # lgb params
    lgb_params = {
            'boosting': 'gbdt',
            'application': 'binary',
            'metric': 'auc', 
            'learning_rate': 0.1,
            'num_leaves': 32,
            'max_depth': 8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'feature_fraction': 0.7,
    }

    # Get folds for k-fold CV
    NFOLD=5
    folds = KFold(n_splits = NFOLD, shuffle = True, random_state = 0)
    fold = folds.split(df)

    eval_score = 0
    n_estimators = 0
    eval_preds = np.zeros(df.shape[0])

    for i, (train_index, test_index) in enumerate(fold):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]

        dtrain = lgb.Dataset(X_train, label = y_train)
        dvalid = lgb.Dataset(X_valid, label = y_valid)

        eval_results = {}

        bst = lgb.train(lgb_params, 
                        dtrain, 
                        valid_sets = [dtrain, dvalid], 
                        valid_names = ['train', 'valid'], 
                        evals_result = eval_results, 
                        num_boost_round = 5000,
                        early_stopping_rounds = 100)
    
        print("\nRounds:", bst.best_iteration)
        print("AUC: ", eval_results['valid']['auc'][bst.best_iteration-1])

        n_estimators += bst.best_iteration
        eval_score += eval_results['valid']['auc'][bst.best_iteration-1]
    
        eval_preds[test_index] += bst.predict(X_valid, num_iteration = bst.best_iteration)

    n_estimators = int(round(n_estimators/NFOLD,0))
    eval_score = round(eval_score/NFOLD,6)

    print("\nModel Report")
    print("Rounds: ", n_estimators)
    print("AUC: ", eval_score)    

    # Get training rows that are most similar to test
    df_av = df.copy()
    df_av['preds'] = eval_preds
    df_av_train = df_av[df_av.target == 0]
    df_av_train = df_av_train.sort_values(by=['preds'], ascending=False).reset_index(drop=True)

    # Check distribution
    # df_av_train.preds.plot()

    # Get valid dataset

    valid_num = int(len(df_av_train) * (1-train_p))
    X_valid = df_av_train.drop(['preds', 'target', 'label'],axis = 1 )[:valid_num].values
    X_train = df_av_train.drop(['preds', 'target', 'label'],axis = 1 )[valid_num:].values
    y_valid = df_av_train['label'][:valid_num].values
    y_train = df_av_train['label'][valid_num:].values

    return X_train, y_train, X_valid, y_valid


def get_adv_feats(df_train, df_test, feats):
    '''
    adv新特征标识训练集测试集
    训练集测试集合并，供后面交叉验证
    '''
    df_train['adv'] = 1
    df_test['adv'] = 0
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    df = df.fillna(0)
    
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': mp.cpu_count()
    }
    
    fold_num = 5
    new_feats = []
    remove_feats = []
    '''
    每个特征依次单独作为训练特征预测adv，
    进行k折交叉验证，
    只要在交叉验证中出现了score超过阈值，
    则说明该特征能很好的区分训练集和测试集，相应的也说明该特征在测试集和训练集上分布差距过大，
    如果训练模型时加入会导致在训练集上过拟合该特征，影响泛化能力，应该剔除。
	'''
    for f in feats:
        X = df[f].values.reshape(-1, 1)
        y =  df['adv'].values
        # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=1, shuffle=True)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        scores = []
        for _fold, (train_idx, val_idx) in enumerate(kf.split(X,y)):
            X_train, X_valid = X[train_idx], X[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]

            model = RandomForestClassifier(max_depth=2, random_state=1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)

            # fpr, tpr, thresholds = roc_curve(y_valid, y_pred, pos_label=2)
            scores.append(roc_auc_score(y_valid, y_pred))

        score = np.mean(scores)
        if f == 'feature62':
            print()
        if score > 0.9:
            remove_feats.append(f)
            print('--------------------------------------', f, score)
        else:
            new_feats.append(f)

    return new_feats, remove_feats


# feats = get_adv_feats(df_train, df_test, feats)
