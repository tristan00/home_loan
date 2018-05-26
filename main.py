import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import gc
import sys
import math

path = 'D:\projects\kaggle\home loan'

params = {
    'num_leaves': 31,
    'objective': 'binary',
    'min_data_in_leaf': 100,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'auc',
    'num_threads': 12,
    'scale_pos_weight':12
}

MAX_ROUNDS = 5000


def split_x_y(df):
    print(df.columns)
    x = df.drop('SK_ID_CURR', axis = 1)
    y = x['TARGET']
    x = x.drop('TARGET', axis = 1)
    return x, y


def get_score(x, y):
    print(x['g_column'].dtype)
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=.1)
    dtrain = lightgbm.Dataset(train_x, label=train_y)
    dval = lightgbm.Dataset(val_x, label=val_y, reference=dtrain)
    model = lightgbm.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
                      verbose_eval=10, categorical_feature='auto')

    columns = x.columns
    f_i = model.feature_importance()
    f1_res = []
    for i, j in zip(columns, f_i):
        f1_res.append({'columns':i, 'f_i':j})
    df =pd.DataFrame.from_dict(f1_res)
    return df


def reduce_data(df, keep_names):
    df['g_column'] = 1

    df_copy = df.copy()
    df = df.drop('TARGET', axis = 1)

    mean_std = df.groupby(['SK_ID_CURR'], as_index=False).mean()
    mean_columns = []

    for i in mean_std.columns:
        if i not in keep_names:
            mean_columns.append(i + '_int_mean')
        else:
            mean_columns.append(i)
    mean_std.columns = mean_columns

    df_copy = df_copy[['SK_ID_CURR', 'g_column', 'TARGET']]
    df_copy = df_copy.merge(mean_std)
    if 'SK_ID_PREV' in df_copy.columns:
        df_copy = df_copy.drop('SK_ID_PREV', axis=1)
    df_copy = df_copy.drop_duplicates()

    if 'g_column' in df_copy.columns:
        df_copy['g_column'] = pd.to_numeric(df_copy['g_column'])
    return df_copy


def preprocess_data(df):
    types = df.dtypes
    les = dict()
    for i, j in zip(types, df.columns):
        print(i, j)

        if str(i) == 'object' and j!= 'SK_ID_CURR' and j!= 'TARGET':
            df[[j]] = df[[j]].fillna('')

            le = LabelEncoder()
            le.fit(df[j])
            les[j]=le
        else:
            df[[j]] = df[[j]].fillna(0)

    for i, j in les.items():
        df[[i]] = j.transform(df[[i]])
    return df


def create_random_function(df):
    pass


def is_top_n_features(df, name, n = 20):
    print(4, name in df['columns'].tolist())
    df = df.sort_values('f_i', ascending=False)
    df = df[:n]
    if name in df['columns'].tolist():
        return True
    else:
        print('failed_feature')
        return False


#CREATED FEATURES
def divide_feature_creator(df, columns, name, df_name, test = True):
    p0 = columns[0]
    p1 = columns[1]
    if test:
        df2 = df.copy()
        df2[name] = df2.apply(lambda x: x[p0] / x[p1] if x[p1] != 0 else x[p0], axis=1)
        # print(1, name in df2.columns)
        # df2 = reduce_data(df2, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        # print(2, name in df2.columns)
        x, y = split_x_y(df2)
        # print(3, name in x.columns)
        f = get_score(x, y)



        if is_top_n_features(f, name):
            # print('here')
            return {'function': 'divide_feature_creator', 'features': columns, 'name': name, 'df_name': df_name}
        else:
            return None
    else:
        df[name] = df.apply(lambda x: x[p0] / x[p1] if x[p1] != 0 else x[p0], axis=1)
        # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        return df


def sum_feature_creator(df, columns, name, df_name, test = True):
    p0 = columns[0]
    p1 = columns[1]
    if test:
        df2 = df.copy()
        df2[name] = df2.apply(lambda x: x[p0] + x[p1], axis=1)
        # print(1, name in df2.columns)
        # df2 = reduce_data(df2, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        # print(2, name in df2.columns)
        x, y = split_x_y(df2)
        # print(3, name in x.columns)
        f = get_score(x, y)

        if is_top_n_features(f, name):
            # print('here')
            return {'function': 'sum_feature_creator', 'features': columns, 'name': name, 'df_name': df_name}
        else:
            return None
    else:
        df[name] = df.apply(lambda x: x[p0] + x[p1], axis=1)
        # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        return df


def std_feature_creator(df, columns, name, df_name, test = True):
    p0 = columns[0]
    p1 = columns[1]
    if test:
        df2 = df.copy()
        print(columns)
        count_df = df2.groupby(p0, as_index=False).std()
        count_df = count_df[columns]
        count_df[name] = count_df[[p1]]
        count_df = count_df.drop(p1, axis = 1)
        df2 = df2.merge(count_df, on = p0)
        # print(1, name in df2.columns)
        # df2 = reduce_data(df2, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        # print(2, name in df2.columns)
        x, y = split_x_y(df2)
        # print(3, name in x.columns)
        f = get_score(x, y)

        if is_top_n_features(f, name):
            # print('here')
            return {'function': 'std_feature_creator', 'features': columns, 'name': name, 'df_name': df_name}
        else:
            return None
    else:
        count_df = df.groupby(p0, as_index=False).std()
        count_df = count_df[columns]
        count_df[name] = count_df[[p1]]
        df = df.merge(count_df, on=p0)
        # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        return df


def mul_feature_creator(df, columns, name, df_name, test = True):
    p0 = columns[0]
    p1 = columns[1]
    if test:
        df2 = df.copy()
        df2[name] = df2.apply(lambda x: x[p0] * x[p1], axis=1)
        # print(1, name in df2.columns)
        # df2 = reduce_data(df2, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        # print(2, name in df2.columns)
        x, y = split_x_y(df2)
        # print(3, name in x.columns)
        f = get_score(x, y)

        if is_top_n_features(f, name):
            # print('here')
            return {'function': 'mul_feature_creator', 'features': columns, 'name': name, 'df_name': df_name}
        else:
            return None
    else:
        df[name] = df.apply(lambda x: x[p0] * x[p1], axis=1)
        # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        return df

def euclidean_feature_creator(df, columns, name, df_name, test = True):
    p0 = columns[0]
    p1 = columns[1]
    if test:
        df2 = df.copy()
        df2[name] = df2.apply(lambda x: math.sqrt((x[p0] * x[p0]) + (x[p1] * x[p1])), axis=1)
        # print(1, name in df2.columns)
        # df2 = reduce_data(df2, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        # print(2, name in df2.columns)
        x, y = split_x_y(df2)
        # print(3, name in x.columns)
        f = get_score(x, y)

        if is_top_n_features(f, name):
            # print('here')
            return {'function': 'euclidean_feature_creator', 'features': columns, 'name': name, 'df_name': df_name}
        else:
            return None
    else:
        df[name] = df.apply(lambda x: math.sqrt((x[p0] * x[p0]) + (x[p1] * x[p1])), axis=1)
        # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        return df


def atan2_feature_creator(df, columns, name, df_name, test = True):
    p0 = columns[0]
    p1 = columns[1]
    if test:
        df2 = df.copy()
        df2[name] = df2.apply(lambda x: math.atan2(x[p0],x[p1]), axis=1)
        # print(1, name in df2.columns)
        # df2 = reduce_data(df2, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        # print(2, name in df2.columns)
        x, y = split_x_y(df2)
        # print(3, name in x.columns)
        f = get_score(x, y)

        if is_top_n_features(f, name):
            # print('here')
            return {'function': 'atan2_feature_creator', 'features': columns, 'name': name, 'df_name': df_name}
        else:
            return None
    else:
        df[name] = df.apply(lambda x: math.atan2(x[p0],x[p1]), axis=1)
        # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        return df


def count_feature_creator(df, columns, name, df_name, test = True):
    if test:
        df2 = df.copy()
        df2[name] = 1
        print(columns)
        count_df = df2.groupby(columns[0], as_index=False).count()
        count_df = count_df[columns + [name]]
        df2 = df2.merge(count_df, on = columns[0])

        # print(1, name in df2.columns)
        # df2 = reduce_data(df2, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        # print(2, name in df2.columns)
        x, y = split_x_y(df2)
        # print(3, name in x.columns)
        f = get_score(x, y)

        if is_top_n_features(f, name):
            # print('here')
            return {'function': 'count_feature_creator', 'features': columns, 'name': name, 'df_name': df_name}
        else:
            return None
    else:
        df[name] = 1
        count_df = df.groupby([columns], as_index=False).count()
        count_df = count_df[columns + [name]]
        df = df.merge(count_df, on = columns)
        # df = reduce_data(df, keep_names=(name, 'SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
        return df



#MANUAL FEATURES
def add_manual_features(df):
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

    df['g_column'] = 1
    return df

#GENETIC ALG
def pick_random_feature(df, name, df_name):
    allowed_functions = ['divide_feature_creator',  'mul_feature_creator', 'sum_feature_creator', 'atan2_feature_creator', 'euclidean_feature_creator']
    # allowed_functions = ['std_feature_creator']

    picked_function = random.choice(allowed_functions)
    columns = random.sample(list(df.columns), 2)
    print(name in df.columns)
    if 'TARGET' in columns or 'SK_ID_CURR' in columns:
        return None
    res =  globals()[picked_function](df, columns, name, df_name)
    print('res', res)
    return res


def explore_features(df1, df2, df2_name):
    added_features = []
    if min(df2.shape) >0:
        df_full = df1.merge(df2, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_{0}'.format(df2_name)))
    else:
        df_full = df1
    df_full = preprocess_data(df_full)

    while len(added_features) < 100:
        res = pick_random_feature(df_full, 'f_{0}_{1}'.format(len(added_features), df2_name), df2_name)
        if res:
            print('feature found', res)
            added_features.append(res)

            globals()[res['function']](df_full, res['features'], res['name'], res['df_name'], test=False)
        print('df columns', len(df_full.columns))
        with open(path + '/features/{0}.plk'.format(df2_name), 'wb') as infile:
            pickle.dump(added_features, infile)


def full_feature_search():
    df_app = pd.read_csv(path + r'/res3.csv')
    # df_app = add_manual_features(df_app)
    explore_features(df_app, pd.DataFrame(), 'df_app')
    return

    df_train = pd.read_csv(path + r'/application_train.csv', nrows=50000)
    df_train = add_manual_features(df_train)
    df_bureau = pd.read_csv(path + r'/bureau.csv')
    df_bureau2 = pd.read_csv(path + r'/bureau_balance.csv')
    df_bureau = df_bureau.merge(df_bureau2, how='left')

    del df_bureau2
    gc.collect()
    explore_features(df_train, df_bureau, 'df_bureau')
    del df_bureau, df_train
    gc.collect()

    df_train = pd.read_csv(path + r'/application_train.csv', nrows=50000)
    df_train = add_manual_features(df_train)
    df_prev = pd.read_csv(path + r'/previous_application.csv')
    explore_features(df_train, df_prev, 'df_prev')
    del df_prev, df_train
    gc.collect()

    df_train = pd.read_csv(path + r'/application_train.csv', nrows=50000)
    df_train = add_manual_features(df_train)
    df_cc = pd.read_csv(path + r'/credit_card_balance.csv')
    explore_features(df_train, df_cc, 'df_cc')
    del df_cc, df_train
    gc.collect()

    df_train = pd.read_csv(path + r'/application_train.csv', nrows=50000)
    df_train = add_manual_features(df_train)
    df_installment = pd.read_csv(path + r'/installments_payments.csv')
    explore_features(df_train, df_installment, 'df_installment')
    del df_installment, df_train
    gc.collect()

    df_train = pd.read_csv(path + r'/application_train.csv', nrows=50000)
    df_train = add_manual_features(df_train)
    df_pos_cash = pd.read_csv(path + r'/POS_CASH_balance.csv')
    explore_features(df_train, df_pos_cash, 'df_pos_cash')
    del df_pos_cash, df_train
    gc.collect()


def execute_features_for_file(df, df2_name):
    with open(path + '/features/{0}.plk'.format(df2_name), 'rb') as infile:
        added_features = pickle.load(infile)

    df = preprocess_data(df)

    print(df.shape)
    print(df.columns.tolist())
    for res in added_features:
        print(res)
        globals()[res['function']](df, res['features'], res['name'], res['df_name'], test=False)
        gc.collect()

    df= reduce_data(df, keep_names=('SK_ID_CURR', 'SK_ID_PREV', 'g_column'))
    return df

def execute_features():
    df_train = pd.read_csv(path + r'/application_train.csv')
    df_test = pd.read_csv(path + r'/application_test.csv')
    df_train = add_manual_features(df_train)
    df_test = add_manual_features(df_test)

    df_total = pd.concat([df_test, df_train])

    # df_total = df_total.sample(n=100)
    print(df_total.shape)

    df_bureau = pd.read_csv(path + r'/bureau.csv')
    df_bureau2 = pd.read_csv(path + r'/bureau_balance.csv')
    df_bureau = df_bureau.merge(df_bureau2, how='outer', suffixes=('', '_2'))
    df_bureau = df_bureau.merge(df_total, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_{0}'.format('df_bureau')))
    del df_bureau2
    gc.collect()
    df_bureau = execute_features_for_file(df_bureau, 'df_bureau')
    merge_cols = set(df_bureau.columns.tolist())
    df_bureau.to_csv(path + '/temp/df_bureau.csv', index = False)
    del df_bureau
    gc.collect()

    df_prev = pd.read_csv(path + r'/previous_application.csv')
    df_prev = df_prev.merge(df_total, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_{0}'.format('df_prev')))
    gc.collect()
    df_prev = execute_features_for_file(df_prev, 'df_prev')
    merge_cols = merge_cols&set(df_prev.columns.tolist())
    df_prev.to_csv(path + '/temp/df_prev.csv', index = False)
    del df_prev
    gc.collect()

    df_cc = pd.read_csv(path + r'/credit_card_balance.csv')
    df_cc = df_cc.merge(df_total, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_{0}'.format('df_cc')))
    df_cc = execute_features_for_file(df_cc, 'df_cc')
    merge_cols = merge_cols & set(df_cc.columns.tolist())
    df_cc.to_csv(path + '/temp/df_cc.csv', index = False)
    del df_cc
    gc.collect()

    df_installment = pd.read_csv(path + r'/installments_payments.csv')
    df_installment = df_installment.merge(df_total, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_{0}'.format('df_installment')))
    df_installment = execute_features_for_file(df_installment, 'df_installment')
    merge_cols = merge_cols & set(df_installment.columns.tolist())
    df_installment.to_csv(path + '/temp/df_installment.csv', index = False)
    del df_installment
    gc.collect()

    df_pos_cash = pd.read_csv(path + r'/POS_CASH_balance.csv')
    df_pos_cash = df_pos_cash.merge(df_total, how = 'left', on = 'SK_ID_CURR', suffixes=('', '_{0}'.format('df_pos_cash')))
    df_pos_cash = execute_features_for_file(df_pos_cash, 'df_pos_cash')
    merge_cols = merge_cols & set(df_pos_cash.columns.tolist())
    df_pos_cash.to_csv(path + '/temp/df_pos_cash.csv', index = False)
    del df_pos_cash
    gc.collect()

    df_bureau = pd.read_csv(path + '/temp/df_bureau.csv')
    df_prev = pd.read_csv(path + '/temp/df_prev.csv')
    df_output = df_bureau.merge(df_prev, on=list(merge_cols), how='outer')
    del df_bureau, df_prev
    gc.collect()

    df_cc = pd.read_csv(path + '/temp/df_cc.csv')
    df_output = df_output.merge(df_cc, on=list(merge_cols), how='outer')
    del df_cc
    gc.collect()

    df_installment = pd.read_csv(path + '/temp/df_installment.csv')
    df_output = df_output.merge(df_installment, on=list(merge_cols), how='outer')
    del df_installment
    gc.collect()

    df_pos_cash = pd.read_csv(path + '/temp/df_pos_cash.csv')
    df_output = df_output.merge(df_pos_cash, on=list(merge_cols), how='outer')
    del df_pos_cash
    gc.collect()

    df_output = df_output.fillna(0)
    return df_output

def train_full_model():
    df_train = pd.read_csv(path + r'/application_train.csv')
    df_test = pd.read_csv(path + r'/application_train.csv')
    df_train_ids = set(df_train['SK_ID_CURR'])
    df_test_ids = set(df_test['SK_ID_CURR'])

    del df_test, df_train
    gc.collect()

    df = execute_features()
    gc.collect()

    df.to_csv('temp.csv', index = False)

    df_train = df[df['SK_ID_CURR'].isin(df_train_ids)]
    df_test = df[df['SK_ID_CURR'].isin(df_test_ids)]

    y = df_train['TARGET']
    x = df_train.drop(['TARGET', 'SK_ID_CURR'], axis = 1)

    res = df_test[['SK_ID_CURR', 'TARGET']]
    df_test= df_test.drop(['TARGET', 'SK_ID_CURR'], axis = 1)

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=.1)
    dtrain = lightgbm.Dataset(train_x, label=train_y)
    dval = lightgbm.Dataset(val_x, label=val_y, reference=dtrain)
    model = lightgbm.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],
                           early_stopping_rounds=100,
                           verbose_eval=10, categorical_feature='auto')

    res['TARGET'] = model.predict(df_test)
    res.to_csv('output.csv', index =False)
    columns = df_train.columns
    f_i = model.feature_importance()
    f1_res = []
    for i, j in zip(columns, f_i):
        f1_res.append({'columns':i, 'f_i':j})
    df =pd.DataFrame.from_dict(f1_res)
    df.to_csv('f1.csv', index=False)


def main():
    full_feature_search()
    # train_full_model()

if __name__ == '__main__':
    main()

    with open(path + '/features/{0}.plk'.format('df_app'), 'rb') as infile:
        added_features = pickle.load(infile)
    df = pd.DataFrame.from_dict(added_features)
    df.to_csv('features_g.csv', index=False)