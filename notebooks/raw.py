# %%
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit import als

# Модель второго уровня
from lightgbm import LGBMClassifier
import catboost as cb

import sys, os

def current_execute_directory():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        print('working in jupyter')
        return globals()['_dh'][0]

current_directory = current_execute_directory()
# Написанные нами функции
additional_functions_path = os.path.join(current_directory, os.pardir)
sys.path.insert(0, additional_functions_path)

from src.metrics import precision_at_k, recall_at_k
from src.utils import prefilter_items, process_user_item_features
from src.recommenders import MainRecommender

import pickle
MODELS_PATH = os.path.join(current_directory, os.pardir, 'models')

# %% [markdown]
# 
# ## Read data

# %%
DATA_PATH = '../data'
data = pd.read_csv(os.path.join(current_directory, DATA_PATH,'retail_train_sample.csv'))
item_features = pd.read_csv(os.path.join(current_directory, DATA_PATH,'product.csv'))
user_features = pd.read_csv(os.path.join(current_directory, DATA_PATH,'hh_demographic.csv'))

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

data = data.drop('Unnamed: 0', axis=1, errors='ignore')

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

# %% [markdown]
# Here is how the fact table looks like:

# %%
data.head()

# %% [markdown]
# And here are the descriptive datasets:

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


data_val_matcher = data_val_matcher.loc[data_val_matcher.user_id.isin(common_users)]
data_train_ranker = data_train_ranker.loc[data_train_ranker.user_id.isin(common_users)]
data_val_ranker = data_val_ranker.loc[data_val_ranker.user_id.isin(common_users)]

print_stats_data(data_train_matcher,'train_matcher')

print_stats_data(data_val_matcher,'val_matcher')
print_stats_data(data_train_ranker,'train_ranker')
print_stats_data(data_val_ranker,'val_ranker')

# %%
result_matcher = data_val_matcher.groupby(USER_COL)[ITEM_COL].unique().reset_index()
result_matcher.columns = [USER_COL, ACTUAL_COL]

result_matcher.head()

# %%
# recommender = MainRecommender(data_train_matcher)

# %%
baseline_path = os.path.join(MODELS_PATH, 'baseline_fitted_model.pkl')
# pickle.dump(recommender, open(baseline_path, 'wb'))

# %%
recommender = pickle.load(open(baseline_path, 'rb'))

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
matcher_users = result_matcher[USER_COL]

# result_matcher['als'] = matcher_users.apply(lambda x: get_recommendations(x, 'als', N_PREDICT))
# result_matcher['own'] = matcher_users.apply(lambda x: get_recommendations(x, 'own', N_PREDICT))
# result_matcher['similar_items'] = matcher_users.apply(lambda x: get_recommendations(x, 'similar_items', N_PREDICT))
# result_matcher['similar_users'] = matcher_users.apply(lambda x: get_recommendations(x, 'similar_users', N_PREDICT))

matcher_result_path = os.path.join(MODELS_PATH, 'result_matcher.pkl')
# result_matcher.to_pickle(matcher_result_path)
result_matcher.head()
result_matcher = pd.read_pickle(matcher_result_path)
# %%
recs = result_matcher.columns.tolist()[2:]

def calc_recall(df_data, columns, top_k):
    for col_name in columns:
        yield col_name, df_data.apply(lambda row: recall_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean()

def calc_precision(df_data, columns, top_k):
    for col_name in columns:
        yield col_name, df_data.apply(lambda row: precision_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean()

print(sorted(calc_recall(result_matcher, recs, 50), key=lambda x: x[1],reverse=True))

# %%
print(sorted(calc_precision(result_matcher, recs, 5), key=lambda x: x[1],reverse=True))

# %% [markdown]
# -------

# %%
# take users from train ranker
df_match_candidates = pd.DataFrame(data_train_ranker[USER_COL].unique())
df_match_candidates.columns = [USER_COL]

# get candidates using model that perfomanced the best
df_match_candidates['candidates'] = df_match_candidates[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))

# %%

# unstack users to items candidates
df_items = df_match_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
df_items.name = ITEM_COL
# join 
df_match_candidates = df_match_candidates.drop('candidates', axis=1).join(df_items)
df_match_candidates.head()

# %%
print_stats_data(data_train_ranker, 'train ranker')
print_stats_data(df_match_candidates, 'match_candidates')

# %%
df_ranker_train = data_train_ranker[[USER_COL, ITEM_COL]].copy()
df_ranker_train['target'] = 1 # only purchases

# merge candidates
df_ranker_train = df_match_candidates.merge(df_ranker_train, on=[USER_COL, ITEM_COL], how='left')
# drop dublicates
df_ranker_train = df_ranker_train.drop_duplicates(subset=[USER_COL, ITEM_COL])

df_ranker_train['target'].fillna(0, inplace= True)

# %%
df_ranker_train.sample(5)

# %%
df_ranker_train.target.value_counts()

# %%
# Generate new features
dt = df_join_train_matcher.copy()
users = user_features.copy()
items = item_features.copy()

users, items = process_user_item_features(dt, users, items)
df_ranker_train = df_ranker_train.merge(users, how='left', on='user_id').merge(items, how='left', on='item_id')
df_ranker_train.head(3)

# %%
def convert_categorical_to_int(series):
    if series.dtypes == 'float64':
        return series.astype('int64')
    else:
        return series
# %%
# define categorical columns and convert them to category inplace
object_cols = df_ranker_train.select_dtypes(include=['object']).columns.tolist()
department_cols = [col for col in df_ranker_train.columns if col in item_features.department.unique().tolist()]
categorical_cols = object_cols + department_cols + ['manufacturer']

df_ranker_train.dropna(inplace=True)           
for col in categorical_cols:
    df_ranker_train[col] = convert_categorical_to_int(df_ranker_train[col])

cat_numeric_cols  = df_ranker_train[categorical_cols].select_dtypes(include='int64').columns.tolist()
cat_object_cols = [col for col in categorical_cols if col not in cat_numeric_cols] # select only non-numeric categorical columns
df_ranker_train[cat_object_cols] = df_ranker_train[cat_object_cols].astype('category')


# %%
ranker_processed_path = os.path.join(MODELS_PATH, 'ranker_train_processed.pkl')
df_ranker_train.to_pickle(ranker_processed_path)

ranker_processed_path = pd.read_pickle(ranker_processed_path)
# %%
X_train = df_ranker_train.drop('target', axis=1)
y_train = df_ranker_train[['target']]
print('X_train', X_train.shape)
print('y_train', y_train.shape)

# %%
X_train.isna().sum()

# %%
import catboost as cb

# train_dataset = cb.Pool(X_train,y_train, 
                        # cat_features=categorical_cols)                                                      
# test_dataset = cb.Pool(X_val,y_val,           
                    #    cat_features=categorical_cols)
                    
# model = cb.CatBoostClassifier(loss_function='Logloss',  
                            #   eval_metric='AUC')
grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5,],
        'iterations': [50, 100, 150]}
# model.grid_search(grid,train_dataset)

cb_model_path = os.path.join(MODELS_PATH, 'catboost_fitted_model.pkl')
# pickle.dump(model, open(cb_model_path, 'wb'))
model = pickle.load(open(cb_model_path, 'rb'))
pred = model.predict_proba(X_train)

# %%
from sklearn.metrics import roc_auc_score, roc_curve
def plot_roc_auc_score(y_test, y_pred):
    auc = roc_auc_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, thresolds = roc_curve(y_test, y_pred)

    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

# %%
from sklearn.metrics import roc_curve
def calc_roc_auc_score():
    pass
model_cols = result_matcher.columns[2:]
k=[1, 2, 5, 10]

recs = result_matcher[model_cols[0]][0][:k[1]]
actual = result_matcher[ACTUAL_COL][0]
tp_fp = np.array([1 if rec in actual else 0 for rec in recs])
tp = tp_fp[tp_fp == 1].size
fp = tp_fp[tp_fp == 0].size
fn_mask = np.array([1 if purchase in recs else 0 for purchase in actual])
fn = fn_mask[fn_mask==0].size
tp, fp, fn



