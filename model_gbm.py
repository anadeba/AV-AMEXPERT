from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd

def categorical_variable_encoding(df):
   return pd.get_dummies(df, columns=list(df))


### GBM - Fix learning rate and number of estimators

parameters = {
                'learning_rate' : [ 0.01, 0.1, 0.5 ],
                'n_estimators': [100],
                'min_samples_leaf' : range(10,50,5),
                'max_depth' : range(2,10)
            }

gbm_clfr = GradientBoostingClassifier(max_features='sqrt',subsample=0.8,random_state=10)
gsearch1 = GridSearchCV(estimator = gbm_clfr, param_grid = parameters, scoring='roc_auc', n_jobs=4,iid=False, cv=5, verbose=5)
gsearch1.fit(df_train_X, df_train_y)
gsearch1.best_params_, gsearch1.best_score_