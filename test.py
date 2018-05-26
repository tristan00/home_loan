import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import lightgbm
from sklearn.model_selection import train_test_split
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.decomposition import PCA
from keras import callbacks
from sklearn.preprocessing import MinMaxScaler
path = 'D:\projects\kaggle\home loan'


params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 100,
    'learning_rate': 0.01,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'auc',
    'num_threads': 12,
    'scale_pos_weight':12
}
MAX_ROUNDS = 5000


class feature():

    def __init__(self, df):
        pass

    def divide(self, ):
        pass

eligible_rows = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'AMT_ANNUITY', 'REGION_POPULATION_RELATIVE',
                 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'EXT_SOURCE_1', 'HOUR_APPR_PROCESS_START', 'AMT_GOODS_PRICE',
                 'ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'WEEKDAY_APPR_PROCESS_START',
                 'OWN_CAR_AGE', 'NAME_FAMILY_STATUS', 'OBS_60_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE',
                 'CNT_FAM_MEMBERS', 'NAME_HOUSING_TYPE']

def divide_feature_creator(df, columns, name):
    p0 = columns[0]
    p1 = columns[1]
    df[name] = df.apply(lambda x: x[p0] / x[p1] if x[p1] != 0 else x[p0], axis=1)
    # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df


def sum_feature_creator(df, columns, name):
    p0 = columns[0]
    p1 = columns[1]
    df[name] = df.apply(lambda x: x[p0] + x[p1], axis=1)
    # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df


def std_feature_creator(df, columns, name):
    p0 = columns[0]
    p1 = columns[1]
    count_df = df.groupby(p0, as_index=False).std()
    count_df = count_df[columns]
    count_df[name] = count_df[[p1]]
    df = df.merge(count_df, on=p0)
    # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df


def mul_feature_creator(df, columns, name):
    p0 = columns[0]
    p1 = columns[1]
    df[name] = df.apply(lambda x: x[p0] * x[p1], axis=1)
    # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df

def euclidean_feature_creator(df, columns, name):
    p0 = columns[0]
    p1 = columns[1]
    df[name] = df.apply(lambda x: math.sqrt((x[p0] * x[p0]) + (x[p1] * x[p1])), axis=1)
    # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df


def atan2_feature_creator(df, columns, name):
    p0 = columns[0]
    p1 = columns[1]
    df[name] = df.apply(lambda x: math.atan2(x[p0],x[p1]), axis=1)
    # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df


def count_feature_creator(df, columns, name):
    df[name] = 1
    count_df = df.groupby([columns], as_index=False).count()
    count_df = count_df[columns + [name]]
    df = df.merge(count_df, on = columns)
    # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df


def preproccess(df):
    df['a1'] =  df.apply(lambda x: x['AMT_INCOME_TOTAL']/max(1, x['AMT_ANNUITY']), axis = 1)
    df['a2'] = df.apply(lambda x: x['AMT_INCOME_TOTAL'] / max(1, x['AMT_CREDIT']), axis = 1)
    df['a3'] = df.apply(lambda x: x['AMT_INCOME_TOTAL'] / max(1, x['AMT_GOODS_PRICE']), axis = 1)
    df['a4'] = df.apply(lambda x: x['AMT_CREDIT'] / max(1, x['AMT_ANNUITY']), axis = 1)
    df['a5'] = df.apply(lambda x: x['AMT_CREDIT'] / max(1, x['AMT_INCOME_TOTAL']), axis = 1)
    df['a6'] = df.apply(lambda x: x['AMT_CREDIT'] / max(1, x['AMT_GOODS_PRICE']), axis = 1)
    df['a7'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / max(1, x['AMT_CREDIT']), axis = 1)
    df['a8'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / max(1, x['AMT_ANNUITY']), axis = 1)
    df['a9'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / max(1, x['AMT_INCOME_TOTAL']), axis = 1)
    df['a10'] = df.apply(lambda x: x['AMT_ANNUITY'] / max(1, x['AMT_CREDIT']), axis = 1)
    df['a11'] = df.apply(lambda x: x['AMT_ANNUITY'] / max(1, x['AMT_GOODS_PRICE']), axis = 1)
    df['a12'] = df.apply(lambda x: x['AMT_ANNUITY'] / max(1, x['AMT_INCOME_TOTAL']), axis = 1)
    df['a13'] = df.apply(lambda x: x['REGION_POPULATION_RELATIVE'] / max(1, x['AMT_INCOME_TOTAL']), axis = 1)
    df['a14'] = df.apply(lambda x: x['REGION_POPULATION_RELATIVE'] / max(1, x['AMT_ANNUITY']), axis = 1)
    df['a15'] = df.apply(lambda x: x['REGION_POPULATION_RELATIVE'] / max(1, x['AMT_GOODS_PRICE']), axis = 1)
    df['a16'] = df.apply(lambda x: x['REGION_POPULATION_RELATIVE'] / max(1, x['AMT_CREDIT']), axis = 1)
    df['a17'] = df.apply(lambda x: x['EXT_SOURCE_1'] / max(1, x['EXT_SOURCE_2']), axis = 1)
    df['a18'] = df.apply(lambda x: x['EXT_SOURCE_1'] / max(1, x['EXT_SOURCE_3']), axis = 1)
    df['a19'] = df.apply(lambda x: x['EXT_SOURCE_2'] / max(1, x['EXT_SOURCE_3']), axis = 1)

    df['a20'] =  df.apply(lambda x: x['FLAG_MOBIL']/max(1, x['a10']), axis = 1)
    df['a21'] =  df.apply(lambda x: x['AMT_GOODS_PRICE']/max(1, x['ELEVATORS_AVG']), axis = 1)
    df['a22'] = df.apply(lambda x: math.hypot(x['LIVINGAREA_MEDI'], x['OCCUPATION_TYPE']), axis = 1)
    df['a23'] = df.apply(lambda x: x['AMT_ANNUITY'] / max(1, x['APARTMENTS_MEDI']), axis = 1)
    df['a24'] = df.apply(lambda x: x['CODE_GENDER'] + x['LIVINGAREA_MEDI'], axis = 1)
    df['a25'] = df.apply(lambda x: x['DAYS_EMPLOYED'] * x['a6'], axis = 1)
    df['a26'] = df.apply(lambda x: x['CODE_GENDER'] + x['EXT_SOURCE_2'], axis = 1)
    df['a27'] = df.apply(lambda x: x['FLAG_OWN_CAR'] + x['YEARS_BEGINEXPLUATATION_AVG'], axis = 1)
    df['a28'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] * x['CNT_FAM_MEMBERS'], axis = 1)
    df['a29'] = df.apply(lambda x: math.atan2(x['REGION_POPULATION_RELATIVE'], x['EXT_SOURCE_2']), axis = 1)
    df['a30'] = df.apply(lambda x: x['LANDAREA_AVG'] + x['NAME_FAMILY_STATUS'], axis = 1)
    df['a31'] = df.apply(lambda x: math.atan2(x['a14'], x['a2']), axis = 1)
    df['a32'] = df.apply(lambda x: x['DAYS_BIRTH'] + x['YEARS_BEGINEXPLUATATION_MODE'], axis = 1)
    df['a33'] = df.apply(lambda x: x['a4'] + x['AMT_REQ_CREDIT_BUREAU_QRT'], axis = 1)
    df['a34'] = df.apply(lambda x: math.atan2(x['EXT_SOURCE_3'], x['DAYS_REGISTRATION']), axis = 1)
    df['a35'] = df.apply(lambda x: math.hypot(x['EXT_SOURCE_1'], x['LANDAREA_MEDI']), axis = 1)
    df['a36'] = df.apply(lambda x: x['a25'] * x['a14'], axis = 1)
    df['a37'] = df.apply(lambda x: x['a34'] + x['AMT_REQ_CREDIT_BUREAU_QRT'], axis = 1)
    df['a38'] = df.apply(lambda x: math.hypot(x['a4'], x['REG_CITY_NOT_WORK_CITY']), axis = 1)
    df['a39'] = df.apply(lambda x: math.hypot(x['a19'], x['a17']), axis = 1)
    df['a40'] = df.apply(lambda x: x['FLAG_OWN_CAR'] + x['a2'], axis = 1)
    df['a41'] = df.apply(lambda x: x['DAYS_BIRTH'] / max(1, x['OWN_CAR_AGE']), axis = 1)
    df['a42'] = df.apply(lambda x: math.hypot(x['a34'], x['REGION_RATING_CLIENT_W_CITY']), axis = 1)
    df['a43'] = df.apply(lambda x: math.atan2(x['a42'], x['NAME_EDUCATION_TYPE']), axis = 1)
    df['a44'] = df.apply(lambda x: x['EXT_SOURCE_3'] / max(1, x['a8']), axis = 1)
    df['a45'] = df.apply(lambda x: math.hypot(x['AMT_REQ_CREDIT_BUREAU_DAY'], x['a10']), axis = 1)
    return df


def get_bureau_features(df):
    types = df.dtypes
    df['g_column'] = 1
    for i, j in zip(types, df.columns):
        if str(i) == 'object':
            df[j] = df[[j]].fillna('')
            for k in set(df[j]):
                df["feature_{0}_{1}".format(j, k)] = df.apply(lambda x: 1 if x[j] == k else 0, axis=1)
        else:
            df[j] = df[[j]].fillna(0)

    mean = df.groupby(['SK_ID_BUREAU'], as_index=False).mean()
    mean_columns = []
    for i in mean.columns:
        if i != 'SK_ID_CURR' and i != 'SK_ID_PREV' and i != 'TARGET':
            mean_columns.append(i + '_mean')
        else:
            mean_columns.append(i)
    mean.columns = mean_columns

    df['feature_app_count'] = 1
    count_df = df.groupby(['SK_ID_BUREAU'], as_index=False).count()
    count_df = count_df[['SK_ID_BUREAU', 'feature_app_count']]

    df = df[['SK_ID_CURR', 'SK_ID_BUREAU']]
    df = df.merge(mean)
    df = df.merge(count_df)
    df = df.drop_duplicates()
    return df

    # columns = [i for i in df_total.columns if i != 'SK_ID_BUREAU']
    df_bureau_copy = df_bureau.copy()
    df_bureau_copy = df_bureau_copy.drop('SK_ID_CURR', axis = 1)
    mean_std = df_bureau_copy.groupby(['SK_ID_BUREAU'], as_index=False).mean()
    mean_columns = []
    for i in mean_std.columns:
        if i != 'SK_ID_BUREAU':
            mean_columns.append(i + '_b_mean')
        else:
            mean_columns.append(i)
    mean_std.columns = mean_columns
    print(mean_std.columns)
    print(df_bureau.columns)
    df_total = df_bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
    df_total = df_total.drop_duplicates()
    df_total = df_total.merge(mean_std)

    print(df_total.shape)
    return df_total


def proccess_and_reduce_file(df):
    types = df.dtypes
    df['g_column'] = 1

    les = dict()
    for i, j in zip(types, df.columns):
        print(i, j)

        if str(i) == 'object' and j!= 'SK_ID_CURR' and j!= 'TARGET':
            df[j] = df[[j]].fillna('')

            le = LabelEncoder()
            le.fit_transform(df[j])
            les[j]=le
        else:
            df[j] = df[[j]].fillna(0)

    for i, j in zip(types, df.columns):
        if str(i) == 'object':
            df[j] = df[[j]].fillna('')
            for k in set(df[j]):
                df["feature_{0}_{1}".format(j, k)] = df.apply(lambda x: 1 if x[j] == k else 0, axis=1)
        else:
            df[j] = df[[j]].fillna(0)

    mean = df.groupby(['SK_ID_CURR'], as_index=False).mean()
    mean_columns = []
    for i in mean.columns:
        if i != 'SK_ID_CURR' and i != 'SK_ID_PREV' and i!= 'TARGET':
            mean_columns.append(i + '_mean')
        else:
            mean_columns.append(i)
    mean.columns = mean_columns

    df['feature_app_count'] = 1
    count_df = df.groupby(['SK_ID_CURR'], as_index=False).count()
    count_df = count_df[['SK_ID_CURR', 'feature_app_count']]

    df = df[['SK_ID_CURR', 'g_column']]
    df = df.merge(mean)
    df = df.merge(count_df)
    if 'SK_ID_PREV' in df.columns:
        df = df.drop('SK_ID_PREV', axis=1)
    df = df.drop_duplicates()
    return df

def reduce_data(df, keep_names):
    df['g_column'] = 1

    df_copy = df.copy()
    if 'TARGET' in df.columns:
        df_copy = df_copy.drop('TARGET', axis = 1)

    mean_std = df.groupby(['SK_ID_CURR'], as_index=False).mean()
    mean_columns = []

    for i in mean_std.columns:
        if i not in keep_names:
            mean_columns.append(i + '_int_mean')
        else:
            mean_columns.append(i)
    mean_std.columns = mean_columns
    if 'TARGET' in df_copy.columns:
        df_copy = df_copy[['SK_ID_CURR', 'g_column', 'TARGET']]
    else:
        df_copy = df_copy[['SK_ID_CURR', 'g_column']]
    df_copy = df_copy.merge(mean_std)
    if 'SK_ID_PREV' in df_copy.columns:
        df_copy = df_copy.drop('SK_ID_PREV', axis=1)
    df_copy = df_copy.drop_duplicates()

    if 'g_column' in df_copy.columns:
        df_copy['g_column'] = pd.to_numeric(df_copy['g_column'])


    return df_copy


def make_nn(df):
    model = Sequential()
    model.add(Dense(2000, input_dim=(df.shape[1], ), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    import time
    import gc
    start_time = time.time()

    df_train = pd.read_csv(path + r'/application_train.csv')
    df_test = pd.read_csv(path + r'/application_test.csv')
    df_labels = df_train[['SK_ID_CURR', 'TARGET']]
    df_train = df_train.drop('TARGET', axis = 1)

    #
    # df_bureau = pd.read_csv(path + r'/bureau.csv')
    # df_bureau2 = pd.read_csv(path + r'/bureau_balance.csv')
    # df_prev = pd.read_csv(path + r'/previous_application.csv')
    # df_cc = pd.read_csv(path + r'/credit_card_balance.csv')
    # df_installment = pd.read_csv(path + r'/installments_payments.csv')
    # df_pos_cash = pd.read_csv(path + r'/POS_CASH_balance.csv')
    # print('files read', time.time() - start_time)
    #
    # df_bureau =df_bureau.merge(df_bureau2, how='left')
    # print('df_bureau 1', time.time() - start_time)
    # df_bureau = get_bureau_features(df_bureau)
    # print('df_bureau 2', time.time() - start_time)

    df_total = pd.concat([df_train, df_test])
    df_total = proccess_and_reduce_file(df_total)
    print('df_train 1', time.time() - start_time)


    #
    # df_train = proccess_and_reduce_file(df_train)
    # print('df_train 1', time.time() - start_time)
    #
    #
    # df_test = proccess_and_reduce_file(df_test)
    # print('df_test 1', time.time() - start_time)
    #
    # df_bureau = proccess_and_reduce_file(df_bureau)
    # print('df_bureau 2', time.time() - start_time)
    #
    # df_prev = proccess_and_reduce_file(df_prev)
    # print('df_prev 1', time.time() - start_time)
    #
    # df_cc = proccess_and_reduce_file(df_cc)
    # print('df_cc 1', time.time() - start_time)
    #
    # df_installment = proccess_and_reduce_file(df_installment)
    # print('df_installment 1', time.time() - start_time)
    #
    # df_pos_cash = proccess_and_reduce_file(df_pos_cash)
    # print('df_pos_cash 1', time.time() - start_time)
    #
    # print(df_total.shape)
    # df_total = df_total.merge(df_bureau, how='left', on='SK_ID_CURR', suffixes=('', 'df_bureau'))
    # print('df_bureau')
    # print(df_total.shape)
    #
    # df_total = df_total.merge(df_prev, how='left', on='SK_ID_CURR', suffixes=('', '_df_prev'))
    # print('df_prev')
    # print(df_total.shape)
    #
    # df_total = df_total.merge(df_cc, how='left', on='SK_ID_CURR', suffixes=('', '_df_cc'))
    # print('df_cc')
    # print(df_total.shape)
    #
    # df_total = df_total.merge(df_installment, how='left', on='SK_ID_CURR', suffixes=('', '_df_installment'))
    # print('df_installment')
    # print(df_total.shape)
    #
    # df_total = df_total.merge(df_pos_cash, how='left', on='SK_ID_CURR', suffixes=('', '_df_pos_cash'))
    # print('df_pos_cash')
    # print(df_total.shape)
    #
    # print(df_total.columns.tolist())

    df_total = df_total.fillna(-1)

    df_train = df_total[df_total['SK_ID_CURR'].isin(df_train['SK_ID_CURR'].tolist())]
    df_test = df_total[df_total['SK_ID_CURR'].isin(df_test['SK_ID_CURR'].tolist())]
    print('joined 1', time.time() - start_time)


    df_train.to_csv(path + '/train.csv', index=False)
    df_test.to_csv(path + '/test.csv', index=False)

    df_train = df_train.merge(df_labels)
    y = df_train['TARGET']
    df_train = df_train.drop('TARGET', axis=1)

    df_train.to_csv(path + '/res2.csv', index=False)
    df_train = df_train.drop('SK_ID_CURR', axis=1)
    y_test_copy = df_test.copy()
    df_test = df_test.drop('SK_ID_CURR', axis=1)

    # p = PCA(n_components=50)
    # df_train = p.fit_transform(df_train)
    # df_test = p.transform(df_test)

    train_x, val_x, train_y, val_y = train_test_split(df_train, y, test_size=.1)
    dtrain = lightgbm.Dataset(train_x, label=train_y)
    dval = lightgbm.Dataset(val_x, label=val_y, reference=dtrain)
    model = lightgbm.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                           early_stopping_rounds=50,
                           verbose_eval=10, categorical_feature='auto')

    y_test_copy['TARGET'] = model.predict(df_test)
    y_test_copy = y_test_copy[['SK_ID_CURR', 'TARGET']]
    y_test_copy.to_csv('output.csv', index=False)
    columns = df_train.columns
    f_i = model.feature_importance()
    f1_res = []
    for i, j in zip(columns, f_i):
        f1_res.append({'columns': i, 'f_i': j})
    df = pd.DataFrame.from_dict(f1_res)
    df.to_csv('f1.csv', index=False)

    # nn = make_nn(df_train)
    # scaler = MinMaxScaler()
    # df_train = scaler.fit_transform(df_train)
    # df_test= scaler.transform(df_train)
    # cb = callbacks.EarlyStopping(monitor='val_loss',
    #                               min_delta=0,
    #                               patience=2,
    #                               verbose=0, mode='auto')
    # class_weight = {0: 1.,
    #                 1: 12.}
    # nn.fit(df_train, y, validation_split=0.1, callbacks=[cb], class_weight=class_weight, epochs=50)
    #
    # y_test_copy['TARGET'] = model.predict(df_test)
    # y_test_copy.to_csv('outpu2t.csv', index=False)


def main_old():
    df_train = pd.read_csv(path + r'/application_train.csv')
    df_test = pd.read_csv(path + r'/application_test.csv')
    df_bureau = pd.read_csv(path + r'/bureau.csv')
    df_bureau2 = pd.read_csv(path + r'/bureau_balance.csv')
    df_prev = pd.read_csv(path + r'/previous_application.csv')
    df_cc =  pd.read_csv(path + r'/credit_card_balance.csv')
    df_installment = pd.read_csv(path + r'/installments_payments.csv')
    df_pos_cash = pd.read_csv(path + r'/POS_CASH_balance.csv')
    df_bureau.merge(df_bureau2, how = 'left')

    print('df_pos_cash')
    print(df_pos_cash.shape)
    df_pos_cash = get_pos_features(df_pos_cash)
    print(df_pos_cash.shape)
    print()
    print('df_installment')
    print(df_installment.shape)
    df_installment = get_installment_features(df_installment)
    print(df_installment.shape)
    print()
    print('df_cc')
    print(df_cc.shape)
    df_cc = get_cc_features(df_cc)
    print(df_cc.shape)
    print()
    print('df_prev')
    print(df_prev.shape)
    df_prev = get_past_loan_features(df_prev)
    print(df_prev.shape)
    print()
    print()
    print('df_bureau')
    print(df_bureau.shape)
    df_bureau = get_bureau_features(df_bureau, df_bureau2)
    print(df_bureau.shape)


    print(df_train.shape)
    print(df_test.shape)
    df_train = df_train.merge(df_bureau, how='left', on='SK_ID_CURR', suffixes=('', 'df_bureau'))
    df_test = df_test.merge(df_bureau, how='left', on='SK_ID_CURR', suffixes=('', 'df_bureau'))
    print('df_bureau')
    print(df_train.shape)
    print(df_test.shape)

    df_train = df_train.merge(df_prev, how='left', on='SK_ID_CURR', suffixes=('', '_df_prev'))
    df_test = df_test.merge(df_prev, how='left', on='SK_ID_CURR', suffixes=('', '_df_prev'))
    print('df_prev')
    print(df_train.shape)
    print(df_test.shape)

    df_train = df_train.merge(df_cc, how='left', on='SK_ID_CURR', suffixes=('', '_df_cc'))
    df_test = df_test.merge(df_cc, how='left', on='SK_ID_CURR', suffixes=('', '_df_cc'))
    print('df_cc')
    print(df_train.shape)
    print(df_test.shape)

    df_train = df_train.merge(df_installment, how='left', on='SK_ID_CURR', suffixes=('', '_df_installment'))
    df_test = df_test.merge(df_installment, how='left', on='SK_ID_CURR', suffixes=('', '_df_installment'))
    print('df_installment')
    print(df_train.shape)
    print(df_test.shape)

    df_train = df_train.merge(df_pos_cash, how='left', on='SK_ID_CURR', suffixes=('', '_df_pos_cash'))
    df_test = df_test.merge(df_pos_cash, how='left', on='SK_ID_CURR', suffixes=('', '_df_pos_cash'))
    print('df_pos_cash')
    print(df_train.shape)
    print(df_test.shape)

    print(df_train.shape)
    print(df_test.shape)
    df_test = df_test.drop('SK_ID_BUREAU', axis =1)
    df_train = df_train.drop('SK_ID_BUREAU', axis=1)
    df_total = pd.concat([df_test, df_train])

    df_total.to_csv(path + 'res1.csv', index =False)

    types = df_total.dtypes

    les = dict()
    df_test['TARGET'] = 0
    for i, j in zip(types, df_total.columns):
        print(i, j)

        if str(i) == 'object':
            df_total[[j]] = df_total[[j]].fillna('')
            df_train[[j]] = df_train[[j]].fillna('')
            df_test[[j]] = df_test[[j]].fillna('')
            le = LabelEncoder()
            le.fit(df_total[j])
            les[j]=le
        else:
            df_total[[j]] = df_total[[j]].fillna(0)
            df_train[[j]] = df_train[[j]].fillna(0)
            df_test[[j]] = df_test[[j]].fillna(0)

    for i, j in les.items():
        df_train[[i]] = j.transform(df_train[[i]])
        df_test[[i]] = j.transform(df_test[[i]])

    df_test = df_test.drop('TARGET', axis = 1)
    df_train = preproccess(df_train)
    df_test = preproccess(df_test)
    df_train.to_csv(path + '/res3.csv', index=False)
    df_train = reduce_data(df_train, keep_names=('SK_ID_CURR', 'TARGET'))
    df_test = reduce_data(df_test, keep_names=('SK_ID_CURR'))
    y = df_train['TARGET']
    df_train = df_train.drop('TARGET', axis = 1)

    df_train.to_csv(path + '/res2.csv', index=False)
    df_train = df_train.drop('SK_ID_CURR', axis = 1)
    y_test_copy = df_test.copy()
    df_test = df_test.drop('SK_ID_CURR', axis = 1)

    # df_train = df_train[eligible_rows]
    # df_test = df_test[eligible_rows]

    train_x, val_x, train_y, val_y = train_test_split(df_train, y, test_size=.1)
    dtrain = lightgbm.Dataset(train_x, label=train_y)
    dval = lightgbm.Dataset(val_x, label=val_y, reference=dtrain)
    model = lightgbm.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
                      verbose_eval=10, categorical_feature='auto')

    y_test_copy['TARGET'] = model.predict(df_test)

    y_test_copy = y_test_copy[['SK_ID_CURR', 'TARGET']]
    # y_test_copy['TARGET'] = clf.predict(df_test)

    y_test_copy.to_csv('output.csv', index =False)

    columns = df_train.columns
    f_i = model.feature_importance()

    f1_res = []
    for i, j in zip(columns, f_i):
        f1_res.append({'columns':i, 'f_i':j})
    df =pd.DataFrame.from_dict(f1_res)
    df.to_csv('f1.csv', index=False)

    print(df_train.shape)


# df_train = pd.read_csv(path + r'/res3.csv')
# df_train = df_train[0:100]
# df_train.to_csv('sample.csv', index=False)
# print(sum(df_train['TARGET'])/len(df_train['TARGET']))

main()

# NAME_CONTRACT_TYPE_le
# CODE_GENDER_le
# FLAG_OWN_CAR_le
# FLAG_OWN_REALTY_le
# CNT_CHILDREN_le
# NAME_TYPE_SUITE_le
# NAME_INCOME_TYPE_le
# NAME_EDUCATION_TYPE_le
# NAME_FAMILY_STATUS_le




