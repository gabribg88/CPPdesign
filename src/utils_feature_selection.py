import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from utils_training import cross_validate
from sklearn.model_selection import KFold

def r2_score_lgb(preds, train_data):
    metric_name = 'r2'
    y_true = train_data.get_label()
    y_pred = preds
    value = r2_score(y_true, y_pred)
    is_higher_better = True
    return metric_name, value, is_higher_better


def compute_relevance(train,
                      test,
                      feature,
                      selected_features,
                      target,
                      num_folds,
                      num_repeats,
                      seed,
                      params,
                      metrics,
                      optimization_fold):
    
    if feature not in selected_features:
        oof_results, test_results = cross_validate(train=train,
                                                   test=test,
                                                   features=selected_features + [feature],
                                                   target=target,
                                                   num_folds=num_folds,
                                                   num_repeats=num_repeats,
                                                   seed=seed,
                                                   params={**params, **{'num_threads':1, 'scale_pos_weight': 0.1, 'num_leaves': 2}},
                                                   threshold=0.5,
                                                   feval=None,
                                                   refit=False,
                                                   compute_oof_importance=False,
                                                   compute_test_importance=False,
                                                   verbose=False)

        tmp1 = pd.concat(oof_results['score']).set_index(pd.Index(range(1,len(oof_results['score'])+1), name='fold').astype(str))
        tmp1.loc['OOF'] = tmp1.mean(0)
        tmp2 = pd.concat(test_results['score']).set_index(pd.Index(['Test_ensemble'], name='fold').astype(str))
        return (feature, pd.concat([tmp1, tmp2]).loc[optimization_fold, metrics].values)
    else:
        return(feature, np.zeros(len(metrics)))
    
    
def compute_redundancy(comb,
                       feature1,
                       feature2,
                       num_folds,
                       seed,
                       params):
    
    if feature1 != feature2:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        custom_cv = []
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(kf.split(comb[feature1], comb[feature2])):
            custom_cv.append((train_fold_idx, valid_fold_idx))

        train_lgb = lgb.Dataset(comb[[feature1]], comb[feature2], feature_name=[feature1], free_raw_data=False, categorical_feature=[]) 

        cv_results = lgb.cv(params={**params, **{'objective': 'regression', 'num_threads':1, 'metric': None}},
                            train_set=train_lgb,
                            folds=custom_cv,
                            #metrics=None,
                            num_boost_round=params['num_iterations'],
                            #stratified=False,
                            #callbacks=callbacks,
                            eval_train_metric=False,
                            return_cvbooster=False,
                            feval=r2_score_lgb
                           )
        redundancy_feat1_feat2 = np.maximum(cv_results['valid r2-mean'][-1], 0.0)

        train_lgb = lgb.Dataset(comb[[feature2]], comb[feature1], feature_name=[feature2], free_raw_data=False, categorical_feature=[]) 

        cv_results = lgb.cv(params={**params, **{'objective': 'regression', 'num_threads':1, 'metric': None}},
                            train_set=train_lgb,
                            folds=custom_cv,
                            #metrics=None,
                            num_boost_round=params['num_iterations'],
                            #stratified=False,
                            #callbacks=callbacks,
                            eval_train_metric=False,
                            return_cvbooster=False,
                            feval=r2_score_lgb
                           )
        redundancy_feat2_feat1 = np.maximum(cv_results['valid r2-mean'][-1], 0.0)

        return (feature2, (redundancy_feat1_feat2+redundancy_feat2_feat1)/2)
    else:
        return (feature2, 1.0)