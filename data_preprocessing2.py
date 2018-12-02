import pandas as pd
import numpy as np
from datetime import datetime

data_processed_train = pd.read_csv(r'X:\Hackathon\AV - AMEXPERT\train_amex\data_processed_train.csv')

data_processed_train['DateTime'] = pd.to_datetime(data_processed_train['DateTime'])
data_processed_train['Dayofweek'] = data_processed_train['DateTime'].apply(lambda x : x.weekday())
data_processed_train['Hourofday'] = data_processed_train['DateTime'].apply(lambda x : x.hour)

data_processed_train.to_csv(r'X:\Hackathon\AV - AMEXPERT\train_amex\data_processed_train2.csv', index=False)