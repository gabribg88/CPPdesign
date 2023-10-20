import os, random
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import reduce
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
def create_folds(train, features, target, num_folds, num_repeats=None, shuffle=True, seed=42):
    folds = []
    if num_repeats is None:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(skf.split(train[features], train[target])):
            folds.append((train_fold_idx, valid_fold_idx))
    else:
        rskf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=seed)
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(rskf.split(train[features], train[target])):
            folds.append((train_fold_idx, valid_fold_idx))
    return folds

def cross_validate(train, test, features, target, num_folds, num_repeats, seed, params, threshold=0.5, feval=None, refit=True, compute_oof_importance=True, compute_test_importance=True, custom_cv=None, verbose=True, log=100):
    if verbose is False:
        log=0
    
    train = train.copy()
    test = test.copy()
    
    assert(np.array_equal(train.index.values, np.arange(train.shape[0])))
    if custom_cv is None:
        custom_cv = create_folds(train=train, features=features, target=target, num_folds=num_folds, num_repeats=num_repeats, shuffle=True, seed=seed)

    train_lgb = lgb.Dataset(train[features], train[target], feature_name=features, free_raw_data=False, categorical_feature=[])
    test_lgb = lgb.Dataset(test[features], test[target], reference=train_lgb)    

    callbacks = [lgb.log_evaluation(period=log, show_stdv=True),
                 lgb.early_stopping(stopping_rounds=params['early_stopping_round'], first_metric_only=False, verbose=verbose)]

    cv_results = lgb.cv(params=params,
                        train_set=train_lgb,
                        folds=custom_cv,
                        metrics=params['metric'],
                        num_boost_round=params['num_iterations'],
                        #stratified=False,
                        callbacks=callbacks,
                        eval_train_metric=True,
                        return_cvbooster=True,
                        feval=feval
                       )
    best_iteration = cv_results['cvbooster'].best_iteration

    oof_results = {'score': [], 'best_iteration': best_iteration, 'models': cv_results['cvbooster'], 'preds': None, 'feature_importance': []}
    test_results = {'score': [], 'best_iteration': best_iteration, 'models': None, 'preds': None, 'feature_importance': []}

    train['preds'] = 0.
    train['preds_proba'] = 0.
    test['preds_ensemble'] = 0.
    test['preds_proba_ensemble'] = 0.

    for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(custom_cv):
        train.loc[valid_fold_idx, 'fold'] = n_fold+1
        train_fold = train.loc[train_fold_idx].copy()
        valid_fold = train.loc[valid_fold_idx].copy()
        model = cv_results['cvbooster'].boosters[n_fold]
        ### OOF prediction
        train.loc[valid_fold_idx, 'preds_proba'] = model.predict(valid_fold[features], num_iteration=best_iteration)
        train.loc[valid_fold_idx, 'preds'] = train.loc[valid_fold_idx, 'preds_proba'].apply(lambda x: x>=threshold).astype(int)
        score_fold = compute_metrics(true=train.loc[valid_fold_idx, target].values,
                                     preds=train.loc[valid_fold_idx, 'preds'].values, 
                                     preds_proba=train.loc[valid_fold_idx, 'preds_proba'].values, 
                                     fold=n_fold+1)
        oof_results['score'].append(score_fold)
        if compute_oof_importance:
            oof_results['feature_importance'].append(model.predict(valid_fold[features], num_iteration=best_iteration, pred_contrib=True))
        ### Test prediction
        test['preds_proba_ensemble'] += model.predict(test[features], num_iteration=best_iteration) / num_folds
        if compute_test_importance:
            test_results['feature_importance'].append(model.predict(test[features], num_iteration=best_iteration, pred_contrib=True))

    oof_results['preds'] = train
    test['preds_ensemble'] = test['preds_proba_ensemble'].apply(lambda x: x>=threshold).astype(int)
    test_results['score'].append(compute_metrics(true=test[target].values,
                                                     preds=test['preds_ensemble'].values, 
                                                     preds_proba=test['preds_proba_ensemble'].values, 
                                                     fold='Test_ensemble'))
    
    if refit:
        train_lgb = lgb.Dataset(train[features], train[target], feature_name=features, free_raw_data=False, categorical_feature=[])
        test_lgb = lgb.Dataset(test[features], test[target], reference=train_lgb)    

        history = dict()
        callbacks = [lgb.log_evaluation(period=log, show_stdv=True),
                     lgb.record_evaluation(history)]
        
        if params['boosting_type'] == 'dart':
            model = lgb.train(params=params,
                              train_set=train_lgb,
                              valid_sets = [train_lgb, test_lgb],
                              callbacks=callbacks,
                              #num_boost_round=best_iteration,
                              feval=feval)
        else:
            model = lgb.train(params=dict(params, **{'num_iterations': int(best_iteration*1.0), 'early_stopping_round': None}),
                              train_set=train_lgb,
                              valid_sets = [train_lgb, test_lgb],
                              callbacks=callbacks,
                              num_boost_round=best_iteration,
                              feval=feval)
        
        test['preds_proba_refit'] = model.predict(test[features])
        test['preds_refit'] = test['preds_proba_refit'].apply(lambda x: x>=threshold).astype(int)
        test_results['score'].append(compute_metrics(true=test[target].values,
                                                 preds=test['preds_refit'].values, 
                                                 preds_proba=test['preds_proba_refit'].values, 
                                                 fold='Test_refit'))
        
        test_results['models'] = model
    
    test_results['preds'] = test
    
    return oof_results, test_results

def evaluate(train, test, oof_results, test_results, features, target, num_folds, num_repeats, seed, threshold=0.5, compute_oof_importance=True, compute_test_importance=True):
    train = train.copy()
    test = test.copy()
    oof_models = oof_results['models']
    oof_best_iteration = oof_results['best_iteration']
    if test_results is not None:
        test_model = test_results['models']
    else:
        test_model = None
    
    assert(np.array_equal(train.index.values, np.arange(train.shape[0])))
    custom_cv = create_folds(train=train, features=features, target=target, num_folds=num_folds, num_repeats=num_repeats, shuffle=True, seed=seed)

    oof_results = {'score': [], 'best_iteration': oof_best_iteration, 'models': oof_models, 'preds': None, 'feature_importance': []}
    test_results = {'score': [], 'best_iteration': oof_best_iteration, 'models': test_model, 'preds': None, 'feature_importance': []}

    train['preds'] = 0.
    train['preds_proba'] = 0.
    test['preds_ensemble'] = 0.
    test['preds_proba_ensemble'] = 0.

    for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(custom_cv):
        train.loc[valid_fold_idx, 'fold'] = n_fold+1
        train_fold = train.loc[train_fold_idx].copy()
        valid_fold = train.loc[valid_fold_idx].copy()
        model = oof_models.boosters[n_fold]
        ### OOF prediction
        train.loc[valid_fold_idx, 'preds_proba'] = model.predict(valid_fold[features], num_iteration=oof_best_iteration)
        train.loc[valid_fold_idx, 'preds'] = train.loc[valid_fold_idx, 'preds_proba'].apply(lambda x: x>=threshold).astype(int)
        score_fold = compute_metrics(true=train.loc[valid_fold_idx, target].values,
                                     preds=train.loc[valid_fold_idx, 'preds'].values, 
                                     preds_proba=train.loc[valid_fold_idx, 'preds_proba'].values, 
                                     fold=n_fold+1)
        oof_results['score'].append(score_fold)
        if compute_oof_importance:
            oof_results['feature_importance'].append(model.predict(valid_fold[features], num_iteration=oof_best_iteration, pred_contrib=True))
        ### Test prediction
        test['preds_proba_ensemble'] += model.predict(test[features], num_iteration=oof_best_iteration) / num_folds
        if compute_test_importance:
            test_results['feature_importance'].append(model.predict(test[features], num_iteration=oof_best_iteration, pred_contrib=True))

    oof_results['preds'] = train
    test['preds_ensemble'] = test['preds_proba_ensemble'].apply(lambda x: x>=threshold).astype(int)
    test_results['score'].append(compute_metrics(true=test[target].values,
                                                     preds=test['preds_ensemble'].values, 
                                                     preds_proba=test['preds_proba_ensemble'].values, 
                                                     fold='Test_ensemble'))
    
    if test_model is not None:
        model = test_model
        
        test['preds_proba_refit'] = model.predict(test[features])
        test['preds_refit'] = test['preds_proba_refit'].apply(lambda x: x>=threshold).astype(int)
        test_results['score'].append(compute_metrics(true=test[target].values,
                                                 preds=test['preds_refit'].values, 
                                                 preds_proba=test['preds_proba_refit'].values, 
                                                 fold='Test_refit'))
    
    test_results['preds'] = test
    
    return oof_results, test_results

def compute_metrics(true, preds, preds_proba, fold):
    res = pd.DataFrame(data=0.0, index=[fold], columns=['AUROC', 'MCC', 'F1', 'Fb05', 'Fb01', 'ACC', 'SN', 'SP'])
    res['AUROC'] = roc_auc_score(true, preds_proba)
    res['MCC'] = matthews_corrcoef(true, preds)
    res['F1'] = f1_score(true, preds)
    res['Fb05'] = fbeta_score(true, preds, beta=0.5)
    res['Fb01'] = fbeta_score(true, preds, beta=0.1)
    res['ACC'] = accuracy_score(true, preds)
    res['SN'] = recall_score(true, preds)
    res['SP'] = recall_score(true, preds, pos_label=0)
    return res

def print_results(oof_results, test_results=None, display_metrics=True, return_metrics=False):  
    tmp1 = pd.concat(oof_results['score']).set_index(pd.Index(range(1,len(oof_results['score'])+1), name='fold').astype(str))
    tmp1.loc['OOF'] = tmp1.mean(0)
    if test_results is not None:
        try:
            tmp2 = pd.concat(test_results['score']).set_index(pd.Index(['Test_ensemble', 'Test_refit'], name='fold').astype(str))
        except:
            tmp2 = pd.concat(test_results['score']).set_index(pd.Index(['Test_ensemble'], name='fold').astype(str))
    if display_metrics:
        if test_results is not None:
            try:
                display(pd.concat([tmp1, tmp2]).loc[['OOF', 'Test_ensemble', 'Test_refit']])
            except:
                display(pd.concat([tmp1, tmp2]).loc[['OOF', 'Test_ensemble']])
        else:
            display(tmp1.loc[['OOF']])
    if return_metrics:
        if test_results is not None:
            return pd.concat([tmp1, tmp2])
        else:
            return tmp1
    
def plot_importance(oof_results, test_results=None, features=None, max_features=None, show=True, return_imps=False):
    oof_feature_importance = oof_results['feature_importance']
    oof_imps = [pd.DataFrame(oof_feature_importance[f], columns=features + ['expected_values'])\
                [features].abs().mean(axis=0).to_frame(name=f'fold{f+1}') for f in range(len(oof_feature_importance))]
    oof_imps = reduce(lambda df1,df2: pd.merge(df1,df2,left_index=True, right_index=True), oof_imps)
    oof_imps = oof_imps.agg(['mean', 'std'], axis=1).sort_values(by='mean', ascending=False)
    if test_results is not None:
        test_feature_importance = test_results['feature_importance']
        test_imps = [pd.DataFrame(test_feature_importance[f], columns=features + ['expected_values'])\
                    [features].abs().mean(axis=0).to_frame(name=f'fold{f+1}') for f in range(len(test_feature_importance))]
        test_imps = reduce(lambda df1,df2: pd.merge(df1,df2,left_index=True, right_index=True), test_imps)
        test_imps = test_imps.agg(['mean', 'std'], axis=1).sort_values(by='mean', ascending=False)
    
    if max_features is not None:
        oof_imps = oof_imps.iloc[:max_features]
        if test_results is not None:
            test_imps = test_imps.iloc[:max_features]
    if show:
        if test_results is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10,5), sharex=True)
            axes[0].barh(y=oof_imps.index.values, width=oof_imps['mean'].values, xerr=oof_imps['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
            axes[0].set_title('OOF Feature Importance')
            axes[0].invert_yaxis()
            axes[0].set_axisbelow(True)
            axes[1].barh(y=test_imps.index.values, width=test_imps['mean'].values, xerr=test_imps['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
            axes[1].set_title('Test Feature Importance')
            axes[1].invert_yaxis()
            axes[1].set_axisbelow(True)
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6,5))
            ax.barh(y=oof_imps.index.values, width=oof_imps['mean'].values, xerr=oof_imps['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
            ax.set_title('OOF Feature Importance')
            ax.invert_yaxis()
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()
    if return_imps:
        if test_results is not None:
            return oof_imps, test_imps
        else:
            return oof_imps
    
    
from sklearn.metrics import matthews_corrcoef

def matthews_corrcoef_lgb(preds, train_data):
    metric_name = 'mcc'
    y_true = train_data.get_label()
    y_pred = (preds > 0.5).astype(int)
    value = matthews_corrcoef(y_true, y_pred)
    is_higher_better = True
    return metric_name, value, is_higher_better

# from numba import jit

# @jit
# def mcc(tp, tn, fp, fn):
#     sup = tp * tn - fp * fn
#     inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
#     if inf==0:
#         return 0
#     else:
#         return sup / np.sqrt(inf)

# @jit
# def eval_mcc(y_true, y_prob, show=False):
#     idx = np.argsort(y_prob)
#     y_true_sort = y_true[idx]
#     n = y_true.shape[0]
#     nump = 1.0 * np.sum(y_true) # number of positive
#     numn = n - nump # number of negative
#     tp = nump
#     tn = 0.0
#     fp = numn
#     fn = 0.0
#     best_mcc = 0.0
#     best_id = -1
#     prev_proba = -1
#     best_proba = -1
#     mccs = np.zeros(n)
#     for i in range(n):
#         # all items with idx < i are predicted negative while others are predicted positive
#         # only evaluate mcc when probability changes
#         proba = y_prob[idx[i]]
#         if proba != prev_proba:
#             prev_proba = proba
#             new_mcc = mcc(tp, tn, fp, fn)
#             if new_mcc >= best_mcc:
#                 best_mcc = new_mcc
#                 best_id = i
#                 best_proba = proba
#         mccs[i] = new_mcc
#         if y_true_sort[i] == 1:
#             tp -= 1.0
#             fn += 1.0
#         else:
#             fp -= 1.0
#             tn += 1.0
#     if show:
#         y_pred = (y_prob >= best_proba).astype(int)
#         score = matthews_corrcoef(y_true, y_pred)
#         print(score, best_mcc)
#         plt.plot(mccs)
#         return best_proba, best_mcc, y_pred
#     else:
#         return best_mcc
    
#best_proba, best_mcc, y_pred = eval_mcc(oof_results['preds'][TARGET].values, oof_results['preds'].preds_proba.values, True)