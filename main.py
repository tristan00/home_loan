import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import time

path = r'D:\projects\kaggle\home loan/'


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


def get_general_features(df, join_feature, name, remove_columns):
    df_columns = df.columns.tolist()
    columns_to_remove = list(set(df_columns) & set(remove_columns))
    print(name, columns_to_remove)
    df_copy = df.drop(columns_to_remove, axis = 1)

    df = df[remove_columns + [join_feature]]

    count_df = df_copy.groupby(join_feature, as_index = False).count()
    average_df = df_copy.groupby(join_feature, as_index = False).mean()
    min_df = df_copy.groupby(join_feature, as_index = False).min()
    max_df = df_copy.groupby(join_feature, as_index = False).max()

    print(set(average_df.columns) - set(df_copy.columns), set(df_copy.columns) - set(average_df.columns))
    count_columns, average_columns, min_columns, max_columns = [], [], [], []
    for i in df_copy.columns.tolist():
        if i == join_feature:
            count_columns.append(i)
            average_columns.append(i)
            min_columns.append(i)
            max_columns.append(i)
        else:
            count_columns.append('{0}_{1}_{2}'.format(i, name, 'count'))
            average_columns.append('{0}_{1}_{2}'.format(i, name, 'average'))
            min_columns.append('{0}_{1}_{2}'.format(i, name, 'min'))
            max_columns.append('{0}_{1}_{2}'.format(i, name, 'max'))

    count_df.columns = count_columns
    average_df.columns = average_columns
    min_df.columns = min_columns
    max_df.columns = max_columns

    res = count_df.merge(average_df)
    res = res.merge(min_df)
    res = res.merge(max_df)
    res = res.groupby(join_feature, as_index = False).mean()
    res = res.merge(df, how = 'outer')
    return res


def main():
    train_df = pd.read_csv(path + '/application_train.csv')
    test_df = pd.read_csv(path + '/application_test.csv')
    prev_df = pd.read_csv(path + '/previous_application.csv')
    bureau_df = pd.read_csv(path + '/bureau.csv')
    bureau_balance_df = pd.read_csv(path + '/bureau_balance.csv')
    credit_card_df = pd.read_csv(path + '/credit_card_balance.csv')
    POS_CASH_df = pd.read_csv(path + '/POS_CASH_balance.csv')
    payments_df = pd.read_csv(path + '/installments_payments.csv')

    start_time = time.time()

    df_concat = pd.concat([train_df, test_df])
    df_concat = to_catgorical_encodings(df_concat)
    print('df_concat cat', df_concat.shape,time.time() - start_time)


    bureau_df = bureau_df.merge(bureau_balance_df, how = 'outer')

    bureau_df = to_catgorical_encodings(bureau_df)
    print('bureau_df cat', bureau_df.shape,time.time() - start_time)
    prev_df = to_catgorical_encodings(prev_df)
    print('prev_df cat', prev_df.shape, time.time() - start_time)

    credit_card_df = to_catgorical_encodings(credit_card_df)
    print('credit_card_df cat', credit_card_df.shape, time.time() - start_time)

    POS_CASH_df = to_catgorical_encodings(POS_CASH_df)
    print('POS_CASH_df cat', POS_CASH_df.shape, time.time() - start_time)

    payments_df = to_catgorical_encodings(payments_df)
    print('bureau_df cat', bureau_df.shape, time.time() - start_time)


    bureau_df = get_general_features(bureau_df, 'SK_ID_BUREAU', 'bureau', ['SK_ID_CURR'])
    print('bureau_df gen', bureau_df.shape, time.time() - start_time)


    credit_card_df = credit_card_df.merge(payments_df, on = 'SK_ID_PREV')
    credit_card_df = get_general_features(credit_card_df, 'SK_ID_PREV', 'cc_installment', ['SK_ID_CURR'])
    print('credit_card_df gen', bureau_df.shape, time.time() - start_time)


    POS_CASH_df = get_general_features(POS_CASH_df, 'SK_ID_PREV', 'pos_cash', ['SK_ID_CURR'])
    print('POS_CASH_df gen', bureau_df.shape, time.time() - start_time)


    prev_df = prev_df.merge(credit_card_df, how = 'outer')
    prev_df = prev_df.merge(POS_CASH_df, how='outer')


    prev_df = get_general_features(prev_df, 'SK_ID_PREV', 'prev', ['SK_ID_CURR'])
    print('prev_df gen', prev_df.shape, time.time() - start_time)


    df_concat = df_concat.merge(bureau_df, how = 'outer')
    df_concat = df_concat.merge(prev_df, how = 'outer')

    train_df = df_concat[df_concat['SK_ID_CURR'].isin(train_df['SK_ID_CURR'])]
    test_df = df_concat[df_concat['SK_ID_CURR'].isin(test_df['SK_ID_CURR'])]

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    train_df.to_csv('test.csv', index = False)

    res_df = test_df[['SK_ID_CURR', 'TARGET']]

    y = train_df['TARGET']

    train_df = train_df.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
    test_df = test_df.drop(['SK_ID_CURR', 'TARGET'], axis = 1)

    train_x, val_x, train_y, val_y = train_test_split(train_df, y, test_size=.1)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
                      verbose_eval=10, categorical_feature='auto')

    res_df['TARGET'] = model.predict(test_df)


if __name__ == '__main__':
    main()