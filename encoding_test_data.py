import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

data_processed_test = pd.read_csv(r'X:\Hackathon\AV - AMEXPERT\train_amex\data_processed_test2.csv')


def check_missing_values(df):
   df2 = pd.DataFrame()
   for field in list(df):
       df2 = df2.append(df[[field]].isnull().sum().reset_index())
   df2[0] = (df2[0]/df.shape[0])*100
   df2.columns = ['Fields', '% of missing values']
   df2.reset_index(drop = True, inplace = True)
   return df2.sort_values(by = '% of missing values', ascending = False)

keep_columns = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'age_level',
                'user_depth', 'number_of_views', 'number_of_interest', 'var_1', 'total_visit','Hourofday', 'Dayofweek', 'is_click']

df = data_processed_test[keep_columns]


### Imputations
df['user_group_id'] = df['user_group_id'].fillna(13) ### Missing value replaced with new cat '13'
df['gender'] = df['gender'].fillna('missing')        ### Missing value replaced with new cat 'missing'
df['age_level'] = df['age_level'].fillna(7)          ### Missing value replaced with new cat '7'
df['user_depth'] = df['user_depth'].fillna(4)        ### Missing value replaced with new cat '4'


scaler = MinMaxScaler()
df['view_cat'] = np.ceil(scaler.fit_transform(df['number_of_views'].reshape(-1,1)))
df['interest_cat'] = np.ceil(scaler.fit_transform(df['number_of_interest'].reshape(-1,1)))
df['total_visit_cat'] = np.ceil(scaler.fit_transform(df['total_visit'].reshape(-1,1)))

keep_columns_4model = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 'user_group_id', 'gender', 'age_level',
                       'user_depth', 'var_1', 'view_cat', 'interest_cat', 'total_visit_cat', 'Hourofday', 'Dayofweek', 'is_click']

df = df[keep_columns_4model]
df_test_X = df.drop(['is_click'], axis = 1)
df_test_y = pd.DataFrame(df['is_click'])

#np.save(r'X:\Hackathon\AV - AMEXPERT\train_amex\xgb\np_train_X', onehot_encode_it(df_train_X, df_train_X))

def onehot_encode_it(df_fit, df_transform):
    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
    ohe.fit(np.array(df_fit))
    return ohe.transform(np.array(df_transform)).toarray()

##### Label encoding

def label_encode_it(temp_df):
    for f in temp_df.columns:
        le = preprocessing.LabelEncoder()
        f_encoded = le.fit_transform(temp_df[f])
        temp_df[f] = f_encoded
    return np.array(temp_df)


np.save(file = r'X:\Hackathon\AV - AMEXPERT\train_amex\numpy_inputs\np_test_X', arr=label_encode_it(df_test_X))
np.save(file = r'X:\Hackathon\AV - AMEXPERT\train_amex\numpy_inputs\np_test_y', arr=label_encode_it(df_test_y))





### Conversion to libffm

def convert_to_ffm(df, type, numerics, categories, features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    nrows = df.shape[0]
    ncolumns = len(features)
    with open('X:\\Hackathon\\AV - AMEXPERT\\train_amex\\' + str(type) + "_ffm.txt", "w") as text_file:

        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow['is_click']))
            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:
                    # For a new field appearing in a training example
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature
                        # For already encoded fields
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"

            print(nrows-r)
            datastring += '\n'
            text_file.write(datastring)


convert_to_ffm(df, type='test', numerics=['total_visit'],
               categories=['product', 'campaign_id', 'webpage_id', 'product_category_1',
                           'user_group_id', 'gender', 'age_level', 'user_depth', 'var_1', 'is_click', 'view_cat', 'interest_cat'],
               features=['product', 'campaign_id', 'webpage_id', 'product_category_1',
                         'user_group_id', 'gender', 'age_level', 'user_depth', 'var_1', 'total_visit', 'view_cat', 'interest_cat'])