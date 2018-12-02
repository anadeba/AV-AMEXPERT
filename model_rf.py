import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV

train_X = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_train_X_res.npy')
train_Y = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_train_y_res.npy')

va_X = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_va_X.npy')
va_Y = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_va_y.npy')

test_X = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_test_X.npy')

def rf_hyperparam_tune(X_train, y_train, nfolds):
    parameters = {
                    'max_depth' : range(2,10),
                    'min_samples_leaf' : range(10,20,5)
                 }

    rf_clfr = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, verbose=2, oob_score=False, class_weight='balanced_subsample')
    grid_search = GridSearchCV(rf_clfr, parameters, scoring='roc_auc', cv=nfolds, verbose=5, n_jobs=-1)
    grid_search.fit(np.array(X_train), np.array(y_train))
    return grid_search.best_params_, grid_search.best_score_


print(rf_hyperparam_tune(df_train_X, df_train_y, nfolds=5))

rf_clfr = RandomForestClassifier(n_estimators=200, min_samples_leaf = 20, max_depth = 8,random_state=0, n_jobs=-1, verbose=2, oob_score=False, class_weight='balanced_subsample')
rf_clfr.fit(train_X, train_Y)
pred = rf_clfr.predict(va_X)
roc_auc_score(va_Y, pred)

pd.DataFrame(rf_clfr.predict(test_X)).to_csv(r'X:\Hackathon\AV - AMEXPERT\train_amex\pred.csv', index=False)

import imblearn
bal_rf_clrf = imblearn.ensemble.BalancedRandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5, min_samples_leaf=20,
                                                 bootstrap=True, oob_score=False, sampling_strategy='auto',
                                                 replacement=False, n_jobs=3, random_state=0, verbose=0, warm_start=False, class_weight=None)