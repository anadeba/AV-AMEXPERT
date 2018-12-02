### Transfer the data train and test text files to linux machine.
#run the below code in python

import numpy as np
import xlearn as xl
from xlearn import write_data_to_xlearn_format

np_train_X = np.load(r'/home/amexdeba/numpy_inputs/np_train_X_res_U.npy')
np_train_y = np.load(r'/home/amexdeba/numpy_inputs/np_train_Y_res_U.npy')

np_va_X = np.load(r'/home/amexdeba/numpy_inputs/np_va_X.npy')
np_va_y = np.load(r'/home/amexdeba/numpy_inputs/np_va_y.npy')

np_test_X = np.load(r'/home/amexdeba/numpy_inputs/np_test_X.npy')
np_test_y = np.load(r'/home/amexdeba/numpy_inputs/np_test_y.npy')

fields=np.array(range(0,101))
write_data_to_xlearn_format(np_train_X, np_train_y, '/home/amexdeba/third_run/data_train', fields=fields)
write_data_to_xlearn_format(np_va_X, np_va_y, '/home/amexdeba/third_run/data_va', fields=fields)
write_data_to_xlearn_format(np_test_X, np_test_y, '/home/amexdeba/third_run/data_test', fields=fields)


ffm_model=xl.create_ffm()
ffm_model.setTrain("/home/amexdeba/third_run/data_train")
ffm_model.setValidate("/home/amexdeba/third_run/data_va")
param={ 'task':'binary',#‘binary’forclassification,‘reg’forRegression
	'k':2,#Sizeoflatentfactor
	'lr':0.1,#LearningrateforGD
	'lambda':0.0002,#L2RegularizationParameter
	'metric':'auc',#Metricformonitoringvalidationsetperformance
	'epoch':50#MaximumnumberofEpochs
      }

ffm_model.fit(param,"/home/amexdeba/third_run/model.out")



# Convert output to 0/1
ffm_model.setTest("/home/amexdeba/third_run/data_test")
ffm_model.setSign()
ffm_model.predict("/home/amexdeba/third_run/model.out", "/home/amexdeba/third_run/output.txt")


