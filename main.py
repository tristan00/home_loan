import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import catboost
from sklearn.preprocessing import LabelEncoder

path = r'/home/tristan/Documents/projects/home_loan'
params = {
    'num_leaves': 512,
    'objective': 'binary',
    'min_data_in_leaf': 50,
    'learning_rate': 0.01,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'auc',
    'num_threads': 12,
    'scale_pos_weight':12
}
MAX_ROUNDS = 5000



def fill_na_encodings(df):


    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.mask(df == 'XNA/XAP', '')
    df = df.mask(df == 'XNA', '')
    # df = df.replace('XAP', np.nan)
    df = df.mask(df == 365243.00, np.nan)

    for i, j in zip(df.dtypes, df.columns):
        if i == 'object':
            df[j] = df[j].fillna('')
        else:
            df[j] = df[j].fillna(df[j].mean())
    return df


def to_categorical_encodings(df, one_hot = False):
    types = df.dtypes

    categorical_features_indices = np.where(df.dtypes == 'object')
    print(categorical_features_indices)

    if one_hot:
        types = df.dtypes

        dummy_dfs = []

        for i, j in zip(types, df.columns):
            if j == 'SK_ID_CURR' or j == 'TARGET' or j == 'SK_ID_PREV' or j == 'SK_ID_BUREAU':
                continue
            if i == 'object':
                print(j)
                temp_df = pd.get_dummies(df[j], dummy_na=True)
                temp_df_columns = temp_df.columns.tolist()
                temp_df_columns = ["feature_{0}_{1}".format(j, k) for k in temp_df_columns]
                temp_df.columns = temp_df_columns

                dummy_dfs.append(temp_df)
                df = df.drop(j, axis=1)
            else:
                pass
                # df[j] = df[[j]].fillna(0)
    else:
        for i, j in zip(types, df.columns):
            if j == 'SK_ID_CURR' or j == 'TARGET' or j == 'SK_ID_PREV':
                continue
            if i == 'object':

                print(j)
                df[j] = df[j].apply(lambda x: str(x))
                le = LabelEncoder()
                df[j] = le.fit_transform(df[j])
            else:
                pass
                # df[j] = df[[j]].fillna(0)
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
    sum_df = df_copy.groupby(join_feature, as_index=False).sum()

    print(set(average_df.columns) - set(df_copy.columns), set(df_copy.columns) - set(average_df.columns))
    count_columns, average_columns, min_columns, max_columns, sum_columns = [], [], [], [], []
    for i in df_copy.columns.tolist():
        if i == join_feature:
            count_columns.append(i)
            average_columns.append(i)
            min_columns.append(i)
            max_columns.append(i)
            sum_columns.append(i)

        else:
            count_columns.append('{0}_{1}_{2}'.format(i, name, 'count'))
            average_columns.append('{0}_{1}_{2}'.format(i, name, 'average'))
            min_columns.append('{0}_{1}_{2}'.format(i, name, 'min'))
            max_columns.append('{0}_{1}_{2}'.format(i, name, 'max'))
            sum_columns.append('{0}_{1}_{2}'.format(i, name, 'sum'))


    count_df.columns = count_columns
    average_df.columns = average_columns
    min_df.columns = min_columns
    max_df.columns = max_columns
    sum_df.columns = sum_columns

    res = count_df.merge(average_df)
    res = res.merge(min_df)
    res = res.merge(max_df)
    res = res.merge(sum_df)
    res = res.groupby(join_feature, as_index = False).mean()
    res = res.merge(df, how = 'outer')
    return res


def get_final_features():
    pass


def get_application_features(df):
    df['AMT_INCOME_TOTAL/AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    df['AMT_GOODS_PRICE/AMT_INCOME_TOTAL'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / max(1, x['AMT_INCOME_TOTAL']), axis=1)
    df['AMT_CREDIT/AMT_INCOME_TOTAL'] = df.apply(lambda x: x['AMT_CREDIT'] / max(1, x['AMT_INCOME_TOTAL']), axis=1)
    df['AMT_GOODS_PRICE/AMT_CREDIT'] = df.apply(lambda x: x['AMT_GOODS_PRICE'] / max(1, x['AMT_CREDIT']), axis=1)
    df['AMT_CREDIT/AMT_ANNUITY'] = df.apply(lambda x: x['AMT_CREDIT'] / max(1, x['AMT_ANNUITY']), axis=1)
    df['CNT_CHILDREN/CNT_FAM_MEMBERS'] = df.apply(lambda x: x['CNT_CHILDREN'] / max(1, x['CNT_FAM_MEMBERS']), axis=1)
    df['DAYS_BIRTH/DAYS_EMPLOYED'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
    df['DAYS_BIRTH/DAYS_REGISTRATION'] = df['DAYS_BIRTH'] - df['DAYS_REGISTRATION']
    df['DAYS_BIRTH/DAYS_ID_PUBLISH'] = df['DAYS_BIRTH'] - df['DAYS_ID_PUBLISH']
    df['DAYS_BIRTH/DAYS_LAST_PHONE_CHANGE'] = df['DAYS_BIRTH'] - df['DAYS_LAST_PHONE_CHANGE']
    df['DEF_60_CNT_SOCIAL_CIRCLE/OBS_60_CNT_SOCIAL_CIRCLE'] = df.apply(
        lambda x: x['DEF_60_CNT_SOCIAL_CIRCLE'] / max(1, x['OBS_60_CNT_SOCIAL_CIRCLE']), axis=1)
    df['DEF_30_CNT_SOCIAL_CIRCLE/OBS_30_CNT_SOCIAL_CIRCLE'] = df.apply(
        lambda x: x['DEF_30_CNT_SOCIAL_CIRCLE'] / max(1, x['OBS_30_CNT_SOCIAL_CIRCLE']), axis=1)

    return df


def get_bureau_features(df):
    pass
    df_copy = df.copy()

    df['credit_active'] = df.apply(lambda x: 1 if x['CREDIT_ACTIVE'] != 'Closed' else 0, axis=1)
    df['AMT_CREDIT_SUM_DEBT/AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM_LIMIT']
    df['AMT_CREDIT_SUM_DEBT/AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM'] / df['AMT_CREDIT_SUM_DEBT']
    df['AMT_CREDIT_SUM_OVERDUE/AMT_CREDIT_SUM_DEBT'] = df['AMT_CREDIT_SUM_OVERDUE'] / df['AMT_CREDIT_SUM_DEBT']


    df_copy['bureau_count_columns'] = 1
    df_copy = df_copy.groupby(['SK_ID_CURR'], as_index=False).count()
    df = df.merge(df_copy, on = ['SK_ID_CURR'], suffixes=('', '_count'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index=False).max()
    df = df.merge(df_copy, on = ['SK_ID_CURR'], suffixes=('', '_max'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index=False).min()
    df = df.merge(df_copy, on = ['SK_ID_CURR'], suffixes=('', '_min'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index=False).std()
    df = df.merge(df_copy, on = ['SK_ID_CURR'], suffixes=('', '_std'))

    return df


def get_cc_features(df):
    df_copy = df.copy()

    df_copy['cc_count_columns'] = 1
    df_copy = df_copy.groupby(['SK_ID_CURR', 'cc_count_columns'], as_index=False).count()
    df = df.merge(df_copy, suffixes=('', '_count'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index=False).max()
    df = df.merge(df_copy, suffixes=('', '_max'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index=False).min()
    df = df.merge(df_copy, suffixes=('', '_min'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index=False).std()
    df = df.merge(df_copy, suffixes=('', '_std'))

    return df

def get_pos_features(df):
    df_copy = df.copy()

    df['pos_cnt_vs_future_instalment'] = df['CNT_INSTALMENT'] - df['CNT_INSTALMENT_FUTURE']

    df_copy['pos_count_columns'] = 1
    df_copy = df_copy.groupby(['SK_ID_CURR', 'pos_count_columns'], as_index = False).count()
    df = df.merge(df_copy, suffixes = ('', '_count'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).max()
    df = df.merge(df_copy, suffixes = ('', '_max'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).min()
    df = df.merge(df_copy, suffixes = ('', '_min'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).std()
    df = df.merge(df_copy, suffixes = ('', '_std'))

    return df

def get_prev_features(df):
    df_copy = df.copy()
    df_copy['prev_count_columns'] = 1
    df_copy = df_copy.groupby(['SK_ID_CURR', 'prev_count_columns'], as_index = False).count()
    df = df.merge(df_copy, suffixes = ('', '_count'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).max()
    df = df.merge(df_copy, suffixes = ('', '_max'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).min()
    df = df.merge(df_copy, suffixes = ('', '_min'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).std()
    df = df.merge(df_copy, suffixes = ('', '_std'))

    df['prev_AMT_CREDIT/AMT_ANNUITY'] =  df['AMT_CREDIT']/df['AMT_ANNUITY']
    df['prev_AMT_CREDIT/AMT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['prev_AMT_ANNUITY/AMT_GOODS_PRICE'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']

    df['prev_DOWN_PAYMENT/AMT_ANNUITY'] = df['AMT_DOWN_PAYMENT'] / df['AMT_ANNUITY']
    df['prev_DOWN_PAYMENT/AMT_CREDIT'] = df['AMT_DOWN_PAYMENT'] / df['AMT_CREDIT']
    df['prev_DOWN_PAYMENT/AMT_ANNUITY'] = df['AMT_DOWN_PAYMENT'] / df['AMT_ANNUITY']
    df['prev_DOWN_PAYMENT/AMT_APPLICATION'] = df['AMT_DOWN_PAYMENT'] /df['AMT_APPLICATION']
    return df


def get_installment_features(df):
    df_copy = df.copy()
    df_copy['inst_count_columns'] = 1
    df_copy = df_copy[['SK_ID_CURR', 'inst_count_columns']].groupby('SK_ID_CURR', as_index = False).count()

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).count()
    df = df.merge(df_copy, suffixes = ('', '_count'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).max()
    df = df.merge(df_copy, suffixes = ('', '_max'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).min()
    df = df.merge(df_copy, suffixes = ('', '_min'))

    df_copy = df_copy.groupby('SK_ID_CURR', as_index = False).std()
    df = df.merge(df_copy, suffixes = ('', '_std'))

    df = df.merge(df_copy)

    df['prev_days_to_first_payment'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
    df['prev_payment_vs_installment'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    return df


def get_combined_prev_features(df_prev, df_pos, df_installments, df_credict_card_balance):
    pass


def drop_bad_columns(df):
    bad_columns = df.columns.tolist()
    df= df.drop([i for i in bad_columns if 'SK_ID_BUREAU' in i or 'SK_ID_PREV' in i], axis = 1)
    return df


def main():
    train_df = pd.read_csv(path + '/application_train.csv')
    test_df = pd.read_csv(path + '/application_test.csv')
    concat_df = pd.concat([train_df, test_df])
    print('concat')
    concat_df = fill_na_encodings(concat_df)
    print('concat na filled')
    concat_df = get_application_features(concat_df)
    print('concat features')
    concat_df = to_categorical_encodings(concat_df)


    cc_df = pd.read_csv(path + '/credit_card_balance.csv')
    cc_df = fill_na_encodings(cc_df)
    cc_df = to_categorical_encodings(cc_df, one_hot=True)
    cc_df = get_cc_features(cc_df)
    cc_df = fill_na_encodings(cc_df)

    # cc_df = get_general_features(cc_df, 'SK_ID_CURR', 'cc', [])
    cc_df = cc_df.groupby('SK_ID_CURR', as_index =False).mean()


    print('cc done')
    pos_df = pd.read_csv(path + '/POS_CASH_balance.csv')
    pos_df = fill_na_encodings(pos_df)
    pos_df= to_categorical_encodings(pos_df, one_hot=True)
    pos_df = get_pos_features(pos_df)
    pos_df = fill_na_encodings(pos_df)

    # pos_df = get_general_features(pos_df, 'SK_ID_CURR', 'pos', [])
    pos_df = pos_df.groupby('SK_ID_CURR', as_index =False).mean()
    print('pos done')


    prev_df = pd.read_csv(path + '/previous_application.csv')
    prev_df = fill_na_encodings(prev_df)
    prev_df= to_categorical_encodings(prev_df, one_hot=True)
    prev_df = get_prev_features(prev_df)

    # prev_df = get_general_features(prev_df, 'SK_ID_CURR', 'pos', [])
    prev_df = prev_df.groupby('SK_ID_CURR', as_index =False).mean()
    print('prev done')


    installments_df = pd.read_csv(path + '/installments_payments.csv')
    installments_df = fill_na_encodings(installments_df)
    installments_df = to_categorical_encodings(installments_df, one_hot=True)
    installments_df = get_installment_features(installments_df)

    # installments_df = get_general_features(installments_df, 'SK_ID_CURR', 'bureau', [])
    installments_df = installments_df.groupby('SK_ID_CURR', as_index =False).mean()
    print('inst done')

    bureau_df = pd.read_csv(path + '/bureau.csv')
    bureau_df = fill_na_encodings(bureau_df)
    bureau_df = get_bureau_features(bureau_df)
    bureau_df= to_categorical_encodings(bureau_df, one_hot=True)
    bureau_df = bureau_df.groupby('SK_ID_CURR', as_index=False).mean()
    print('bureau done')

    bureau_balance_df = pd.read_csv(path + '/bureau_balance.csv')
    bureau_balance_df = fill_na_encodings(bureau_balance_df)
    bureau_balance_df= to_categorical_encodings(bureau_balance_df, one_hot=True)
    bureau_balance_df = bureau_balance_df.groupby('SK_ID_BUREAU', as_index=False).mean()
    print('bureau done')


    # bureau_balance_df = pd.read_csv(path + '/bureau_balance.csv')
    # bureau_df = bureau_df.merge(bureau_balance_df, how = 'outer

    bureau_df = bureau_df.merge(bureau_balance_df, how = 'left', on = 'SK_ID_BUREAU', suffixes = ('', 'bureau_balance_df_'))

    concat_df = concat_df.merge(cc_df, how = 'left', on = 'SK_ID_CURR', suffixes = ('concat_', 'cc_'))
    concat_df = concat_df.merge(pos_df, how='left', on = 'SK_ID_CURR', suffixes = ('cc_', 'pos_'))
    concat_df = concat_df.merge(prev_df, how='left', on = 'SK_ID_CURR', suffixes = ('pos_', 'prev_'))
    concat_df = concat_df.merge(installments_df, how='left', on = 'SK_ID_CURR', suffixes = ('prev_', 'inst_'))
    concat_df = concat_df.merge(bureau_df, how='left', on = 'SK_ID_CURR', suffixes = ('inst_', 'bureau_'))
    concat_df = fill_na_encodings(concat_df)

    concat_df = concat_df.groupby('SK_ID_CURR', as_index =False).mean()
    concat_df = drop_bad_columns(concat_df)

    train_df = concat_df[concat_df['SK_ID_CURR'].isin(train_df['SK_ID_CURR'])]
    test_df = concat_df[concat_df['SK_ID_CURR'].isin(test_df['SK_ID_CURR'])]

    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)
    print(train_df.shape)
    sample = train_df[0:1000]
    sample.to_csv('sample.csv', index = False)

    res_df = test_df[['SK_ID_CURR', 'TARGET']]
    y = train_df['TARGET']

    train_df = train_df.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
    test_df = test_df.drop(['SK_ID_CURR', 'TARGET'], axis = 1)

    train_x, val_x, train_y, val_y = train_test_split(train_df, y, test_size=.05)
    # dtrain = lgb.Dataset(train_x, label=train_y)
    # dval = lgb.Dataset(val_x, label=val_y, reference=dtrain)
    # model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
    #                   verbose_eval=1, categorical_feature='auto')

    model = catboost.CatBoostClassifier(eval_metric='AUC', n_estimators=10000)
    model.fit(train_x, train_y, eval_set=(val_x, val_y))

    res_df['TARGET'] = model.predict(test_df)

    columns = train_df.columns
    f_i = model.feature_importances_
    f1_res = []
    for i, j in zip(columns, f_i):
        f1_res.append({'columns': i, 'f_i': j})
    df = pd.DataFrame.from_dict(f1_res)
    df.to_csv('f1.csv', index=False)


if __name__ == '__main__':
    main()
    df = pd.read_csv('sample.csv')
    a = 3