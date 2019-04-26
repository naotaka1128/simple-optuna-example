import re
import sys
import optuna
import numpy as np
import pandas as pd
import random
import warnings
from functools import partial
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils.load_data import load_data
warnings.filterwarnings("ignore")

scale_type = False
test_mode = False

def objective(X_train, y_train, df_train, trial):
    boosting_type = trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss'])
    param = {
        'objective': 'binary', 'metric': 'auc',
        'verbosity': -1, 'learning_rate': 0.1,
        'boosting_type': boosting_type,
        'num_leaves': trial.suggest_int('num_leaves', 10, 5000),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 10000),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.0, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 20),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.0, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 1e2),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-3, 1e2),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e2),
    }

    if param['boosting_type'] == 'dart':
        # dart: early_stopping すると精度下がる可能性あるから注意 / ver によっては動かない
        param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if param['boosting_type'] == 'goss':
        param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])

    oof = np.zeros(len(X_train))
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold_, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):
        # 超巨大データで最適化遅ければ捨てても良い
        # if fold_ > 0:
        #     continue

        # TODO: Malwareコンペだけの特殊な設定なのであとで消す
        random.seed(42)
        matcher = ''.join(random.sample('0123456789abcdef', 8))
        adv_sampler = \
            (
                (df_train['AvSigVersion_1'] != 275) & \
                (df_train['MachineIdentifier'].str.match('.+[{}]$'.format(matcher))) & \
                (df_train['test_probability'] > 0.1)
            ) | \
            (df_train['AvSigVersion_1'] == 275)

        train_data = lgb.Dataset(
            X_train.iloc[train_idx],
            label=y_train.iloc[train_idx])
        valid_data = lgb.Dataset(
            X_train.iloc[valid_idx][adv_sampler],
            label=y_train.iloc[valid_idx][adv_sampler])

        clf = lgb.train(param, train_data, num_boost_round=2000,
                        valid_sets = [train_data, valid_data],
                        verbose_eval=100, early_stopping_rounds=50)
        oof[valid_idx] += clf.predict(
            X_train.iloc[valid_idx], num_iteration=clf.best_iteration)

    return -1 * roc_auc_score(y_train[adv_sampler], oof[adv_sampler])


def main(study_name, is_resume_study):
    df_train, _ = load_data(scaled=False)
    df_train = df_train.loc[:, sorted(df_train.columns)]
    X_train = df_train.drop(
        ['HasDetections', 'MachineIdentifier', 'machine_id',
         'AvSigVersion_1', 'test_probability'], axis=1)
    y_train = df_train['HasDetections']

    f = partial(objective, X_train, y_train, df_train)
    if is_resume_study:
        study = optuna.Study(
            study_name=study_name, storage='sqlite:///example.db')
    else:
        study = optuna.create_study(
            study_name=study_name, storage='sqlite:///example.db')
    study.optimize(f, n_trials=50)
    print('params:', study.best_params)


if __name__ == '__main__':
    print('################################## Start Optuna')
    study_name = sys.argv[1]
    is_resume_study = False
    main(study_name, is_resume_study)
