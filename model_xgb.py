from __future__ import division
import numpy as np
import xgboost as xgb
import math




train_X = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_train_X_res_U.npy')
train_Y = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_train_y_res_U.npy')

va_X = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_va_X.npy')
va_Y = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_va_y.npy')
va_Y = va_Y.ravel()

test_X = np.load(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_test_X.npy')



# rescale weight to make it same as test set
weight = train_X[:,100] * float(test_X.shape[0]) / train_Y.shape[0]


sum_wpos = sum( weight[i] for i in range(train_Y.shape[0]) if train_Y[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(train_Y.shape[0]) if train_Y[i] == 0.0  )

# print weight statistics
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))



#### Dmatrix Train, Validation and Test

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_va = xgb.DMatrix(va_X, label=va_Y)
xg_test = xgb.DMatrix(test_X)


# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 3
param['silent'] = 0
param['nthread'] = 3
#param['num_class'] = 2
param['subsample'] = 0.8
param['scale_pos_weight'] = sum_wpos/sum_wneg


"""
# define the preprocessing function
# used to return the preprocessed training, test data, and parameter
# we can use this to do weight rescale, etc.
# as a example, we try to set scale_pos_weight
def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    try:
        wtrain *= sum_weight / sum(wtrain)
        wtest *= sum_weight / sum(wtest)
    except:
        wtrain = wtrain
        wtest = wtest

    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    return (dtrain, dtest, param)

# do cross validation, for each fold
# the dtrain, dtest, param will be passed into fpreproc
# then the return value of fpreproc will be used to generate
# results of that fold
num_round = 100
xgb.cv(param, xg_va, num_round, nfold=5,
       metrics={'auc'}, seed = 0, fpreproc = fpreproc)

"""

##### Boosting Model #######
# you can directly throw param in, though we want to watch multiple metrics here
#plst = list(param.items())+[('eval_metric', 'ams@0.15')]
watchlist = [(xg_train, 'train'), (xg_va, 'test')]
num_round = 200
bst = xgb.train(param, xg_train, num_round, watchlist)


### Sigmoid function

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# get prediction
pred = bst.predict(xg_va)
pred_prob = [sigmoid(x) for x in pred]
pred_label = [1 if x > 0.5 else 0 for x in pred_prob ]
error_rate = np.sum(pred != va_Y) / va_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))