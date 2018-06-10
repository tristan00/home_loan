import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split


path = r'C:\Users\trist\Documents\db_loc\home_loan\files'


params = {
    'num_leaves': 1023,
    'objective': 'regression',
    'min_data_in_leaf': 100,
    'learning_rate': 0.01,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'auc',
    'num_threads': 12
}
MAX_ROUNDS = 10000

df_app_train = pd.read_csv(path + '/application_train.csv')
# df_app_test = pd.read_csv(path + '/application_test.csv')
# df_concat = pd.concat([df_app_test, df_app_train])


x = df_app_train.drop('TARGET', axis = 1)
y = df_app_train['TARGET'].values


def to_catgorical_encodings(df):
    types = df.dtypes
    for i, j in zip(types, df.columns):
        if j == 'SK_ID_CURR' or j == 'TARGET' or j == 'SK_ID_PREV':
            continue
        if i == 'object':
            df[j] = df[[j]].fillna('')
            for k in set(df[j]):
                df["feature_{0}_{1}".format(j, k)] = df.apply(lambda x: 1 if x[j] == k else 0, axis=1)
            df = df.drop(j, axis = 1)
        else:
            df[j] = df[[j]].fillna(0)
    return df

x = to_catgorical_encodings(x)
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=.9)

dtrain = lightgbm.Dataset(x_train, label=y_train)
dval = lightgbm.Dataset(y_train, label=y_val, reference=dtrain)
model = lightgbm.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                       early_stopping_rounds=100,
                       verbose_eval=10)

columns = x.columns
f_i = model.feature_importance()
f1_res = []
for i, j in zip(columns, f_i):
    f1_res.append({'columns': i, 'f_i': j})
df = pd.DataFrame.from_dict(f1_res)
df.to_csv('f1.csv', index=False)




#
#
# groups = []
# for i in df_app_train.columns.tolist():
#     print(i)
#     print(i == 'SK_ID_CURR')
#     if i == 'TARGET' or i == 'SK_ID_CURR':
#         print('here')
#     else:
#         groups.append({i: df_app_train[[i, 'TARGET']].groupby([i]).agg(['mean', 'count'])})
#
# df_g = df_concat.groupby('')
# df_bureau = pd.read_csv(path + '/bureau.csv')
# df_concat = df_concat.merge(df_bureau, how = 'left', on= 'SK_ID_CURR')
# df_concat = df_concat.dropna(subset = ['SK_ID_BUREAU'])
# df_concat = df_concat[:100]
# df_bureau_balance = pd.read_csv(path + '/bureau_balance.csv')
# # df_cc = pd.read_csv(path + '/credit_card_balance.csv')
# # df_installment = pd.read_csv(path + '/installments_payments.csv')
# # df_pos_cash = pd.read_csv(path + '/POS_CASH_balance.csv')
# # df_prev = pd.read_csv(path + '/previous_application.csv')
#
#
# df_burau_joined = df_bureau.merge(df_bureau_balance, on = 'SK_ID_BUREAU')
# print(1)