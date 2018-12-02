import pandas as pd
import numpy as np

data = pd.read_csv(r'X:\Hackathon\AV - AMEXPERT\test_LNMuIYp\test.csv')
hist_user_logs = pd.read_csv(r'X:\Hackathon\AV - AMEXPERT\train_amex\historical_user_logs.csv')


def check_missing_values(df):
   df2 = pd.DataFrame()
   for field in list(df):
       df2 = df2.append(df[[field]].isnull().sum().reset_index())
   df2[0] = (df2[0]/df.shape[0])*100
   df2.columns = ['Fields', '% of missing values']
   df2.reset_index(drop = True, inplace = True)
   return df2.sort_values(by = '% of missing values', ascending = False)


user_list = data['user_id'].unique()
temp_df = pd.DataFrame()

#user_list = user_list[0:10]
for usr in user_list:
    usr_data = data[data['user_id'] == usr]
    usr_hist_user_logs = hist_user_logs[hist_user_logs['user_id'] == usr]
    usr_hist_total_visit = len(usr_hist_user_logs)

    for prdct in list(usr_data['product'].unique()):
        number_of_views = np.sum((usr_hist_user_logs['product'] == prdct) & (usr_hist_user_logs['action'] == 'view'))
        number_of_interest = np.sum((usr_hist_user_logs['product'] == prdct) & (usr_hist_user_logs['action'] == 'interest'))
        temp_df = temp_df.append(pd.DataFrame([[usr,prdct,number_of_views,number_of_interest,usr_hist_total_visit]], columns=['user_id','product','number_of_views','number_of_interest', 'total_visit']))

    print(np.where(user_list == usr))

data_processed = data.merge(temp_df, how = 'left', on=['user_id', 'product'])

#temp_df.to_csv(r'X:\Hackathon\AV - AMEXPERT\train_amex\user_behavious.csv', index=False)
data_processed.to_csv(r'X:\Hackathon\AV - AMEXPERT\train_amex\data_processed_test.csv', index=False)