import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


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

def to_categorical_encodings(df):
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
            df = df.drop(j, axis = 1)
        else:
            df[j] = df[[j]].fillna(0)

    df = pd.concat([df] + dummy_dfs, axis = 1)
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


def fillna():
    pass


def main():
    train_df = pd.read_csv(path + '/application_train.csv')
    test_df = pd.read_csv(path + '/application_test.csv')
    concat_df = pd.concat([train_df, test_df])

    bureau_df = pd.read_csv(path + '/bureau.csv')
    bureau_balance_df = pd.read_csv(path + '/bureau_balance.csv')
    bureau_df = bureau_df.merge(bureau_balance_df, how = 'outer')

    bureau_df = to_categorical_encodings(bureau_df)
    bureau_df = get_general_features(bureau_df, 'SK_ID_BUREAU', 'bureau', ['SK_ID_CURR'])
    bureau_df = bureau_df.groupby('SK_ID_CURR', as_index =False).mean()
    concat_df = to_categorical_encodings(concat_df)

    concat_df = concat_df.merge(bureau_df, how = 'left')
    concat_df = concat_df.groupby('SK_ID_CURR', as_index =False).mean()

    train_df = concat_df[concat_df['SK_ID_CURR'].isin(train_df['SK_ID_CURR'])]
    test_df = concat_df[concat_df['SK_ID_CURR'].isin(test_df['SK_ID_CURR'])]

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    res_df = test_df[['SK_ID_CURR', 'TARGET']]
    y = train_df['TARGET']

    train_df = train_df.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
    test_df = test_df.drop(['SK_ID_CURR', 'TARGET'], axis = 1)

    train_x, val_x, train_y, val_y = train_test_split(train_df, y, test_size=.1)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dtrain, dval],early_stopping_rounds=50,
                      verbose_eval=10, categorical_feature='auto')

    columns = train_df.columns
    f_i = model.feature_importance()
    f1_res = []
    for i, j in zip(columns, f_i):
        f1_res.append({'columns': i, 'f_i': j})
    df = pd.DataFrame.from_dict(f1_res)
    df.to_csv('f1.csv', index=False)


if __name__ == '__main__':
    main()