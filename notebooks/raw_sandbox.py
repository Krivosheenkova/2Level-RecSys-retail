# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit import als

# Модель второго уровня
from lightgbm import LGBMClassifier

import os, sys
module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Написанные нами функции
from src.metrics import precision_at_k, recall_at_k
from src.utils import prefilter_items
from src.recommenders import MainRecommender

# %%
"""
## Read data
"""

# %%
DATA_PATH = '../data'
data = pd.read_csv(os.path.join(DATA_PATH,'retail_train.csv'))
item_features = pd.read_csv(os.path.join(DATA_PATH,'product.csv'))
user_features = pd.read_csv(os.path.join(DATA_PATH,'hh_demographic.csv'))

# %%
ITEM_COL = 'item_id'
USER_COL = 'user_id'
ACTUAL_COL = 'actual'

# N = Neighbors
N_PREDICT = 50 

# %%
# column processing
item_features.columns = [col.lower() for col in item_features.columns]
user_features.columns = [col.lower() for col in user_features.columns]

item_features.rename(columns={'product_id': ITEM_COL}, inplace=True)
user_features.rename(columns={'household_key': USER_COL }, inplace=True)

# %%
VAL_MATCHER_WEEKS = 8
VAL_RANKER_WEEKS = 3

data_train_matcher = data[data['week_no'] < data['week_no'].max() - (VAL_MATCHER_WEEKS + VAL_RANKER_WEEKS)] # давние покупки
data_val_matcher = data[(data['week_no'] >= data['week_no'].max() - (VAL_MATCHER_WEEKS + VAL_RANKER_WEEKS)) &
                      (data['week_no'] < data['week_no'].max() - (VAL_RANKER_WEEKS))]

data_train_ranker = data_val_matcher.copy()  # Для наглядности. Далее мы добавим изменения, и они будут отличаться
data_val_ranker = data[data['week_no'] >= data['week_no'].max() - VAL_RANKER_WEEKS]

# ----
print('data_train_matcher: {}-{} weeks'.format(data_train_matcher.week_no.min(), data_train_matcher.week_no.max()))
print('data_val_matcher: {}-{} weeks'.format(data_val_matcher.week_no.min(), data_val_matcher.week_no.max()))

print('data_train_ranker: {}-{} weeks'.format(data_train_ranker.week_no.min(), data_train_ranker.week_no.max()))
print('data_val_ranker: {}-{} weeks'.format(data_val_ranker.week_no.min(), data_val_ranker.week_no.max()))

# %%
# сделаем объединенный сет данных для первого уровня (матчинга)
df_join_train_matcher = pd.concat([data_train_matcher, data_val_matcher])

# %%
test_size_weeks = 3

data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

print('data_train shape: %d\tdata_test shape: %d' % (data_train.shape[0], data_test.shape[0]))

# %%
pd.set_option('display.max_columns', None)

# %%
result_matcher = data_test.groupby('user_id').item_id.unique().reset_index().rename(columns={'item_id': 'actual'})

result_matcher.head()

# %%
test_users = result_matcher.shape[0]
new_test_users = len(set(data_test['user_id']) - set(data_train['user_id']))

print('There are {} users in test dataset'.format(test_users))
print('here are {} new users in test dataset'.format(new_test_users))

# %%
"""
Here is how the fact table looks like:
"""

# %%
data.head()

# %%
"""
And here are the descriptive datasets:
"""

# %%
item_features.head()

# %%
user_features.head()

# %%
data_train_matcher = prefilter_items(data=data_train_matcher
                                    , item_features=item_features
                                    , take_n_popular=5000)

# %%
def print_stats_data(data, name):
    print(name)
    print(f'shape: {data.shape}\titems: {data[ITEM_COL].nunique()}\tusers: {data[USER_COL].nunique()}')

# %%
# make cold-start warm
common_users = data_train_matcher.user_id.values

data_val_matcher = data_val_matcher[data_val_matcher.user_id.isin(common_users)]
data_train_ranker = data_train_ranker[data_train_ranker.user_id.isin(common_users)]
data_val_ranker = data_val_ranker[data_val_ranker.user_id.isin(common_users)]

print_stats_data(data_train_matcher,'train_matcher')
print_stats_data(data_val_matcher,'val_matcher')
print_stats_data(data_train_ranker,'train_ranker')
print_stats_data(data_val_ranker,'val_ranker')

# %%
recommender = MainRecommender(data_train_matcher)

# %%
def get_recommendations(user, model, N):
    if model == 'als':
        return recommender.get_als_recommendations(user, N=N)
    elif model == 'own':
        return recommender.get_own_recommendations(user, N=N)
    elif model == 'similar_items':
        return recommender.get_similar_items_recommendation(user, N=N)
    elif model == 'similar_users':
        return recommender.get_similar_users_recommendation(user, N=N)

# %%
result_matcher['als'] = result_matcher['user_id'].apply(lambda x: get_recommendations(x, 'als', N_PREDICT))
result_matcher['own'] = result_matcher['user_id'].apply(lambda x: get_recommendations(x, 'own', N_PREDICT))
result_matcher['similar_items'] = result_matcher['user_id'].apply(lambda x: get_recommendations(x, 'similar_items', N_PREDICT))
result_matcher['similar_users'] = result_matcher['user_id'].apply(lambda x: get_recommendations(x, 'similar_users', N_PREDICT))

result_matcher.head()

# %%
result_matcher.loc[result_matcher.index == result_matcher.index.max()]