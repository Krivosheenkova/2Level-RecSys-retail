# utils.py
from typing import List
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def prefilter_items(data, item_features, take_n_popular=5000):

    print('== Starting prefilter info ==')
    n_users = data.user_id.nunique()
    n_items = data.item_id.nunique()
    sparsity = float(data.shape[0]) / float(n_users*n_items) * 100
    print('shape: {}'.format(data.shape))
    print('# users: {}'.format(n_users))
    print('# items: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    data_train = data.copy()

    # do not use top popular items (they'd be bought anyway)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / data_train.user_id.nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    top_popular = popularity[popularity['share_unique_users'] > .5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # do not use top not popular
    top_not_popular = popularity[popularity.share_unique_users < .0009].item_id.tolist()
    data = data[~data.item_id.isin(top_not_popular)]

    # do not use items that have not been sold in the last 12 month
    num_weeks = 12*4
    start_week = data_train.week_no.max() - num_weeks
    items_sold_last_year = data[data.week_no >= start_week].item_id.tolist()
    data = data[data.item_id.isin(items_sold_last_year)]

    # do not use not popular departments
    merged_data_departments = data_train[['user_id', 'item_id', 'quantity']].merge(item_features[['item_id', 'department']], how='left')
    quantity_by_department = merged_data_departments.groupby('department')['quantity'].sum().reset_index()
    quantity_by_department['coef'] = quantity_by_department.quantity / quantity_by_department.quantity.sum()
    not_popular_departments = quantity_by_department[quantity_by_department.coef < quantity_by_department.coef.quantile(0.25)].department.tolist()
    
    not_popular_departments_items = item_features[
        item_features.department.isin(not_popular_departments)].item_id.tolist()
    data = data[~data.item_id.isin(not_popular_departments_items)]

    # do not use too expensive and too cheap items
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    high_cost_threshold = 2  # mailing cost 2$ 
    low_cost_threshold = data_train.sales_value.quantile(.11)
    data = data[
        (data.sales_value < high_cost_threshold)
        &
        (data.sales_value > low_cost_threshold)
        ]    

    # do not use too popular stores
    store_df = data.groupby('store_id')['user_id'].nunique().reset_index()
    data = data[~data.store_id.isin(
        store_df[store_df.user_id > store_df.user_id.quantile(.985)].store_id.tolist()
    )]

    # Take n top popularity
    popularity = data.groupby('item_id')['quantity'].sum().reset_index(name='n_sold')
    # popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    # Insert fake item_id, if user have bought from top then user have been "bought" this item already
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    # take n poplar items
    if take_n_popular:
        popularity = data.groupby('item_id')['user_id'].nunique().reset_index().sort_values('user_id', ascending=False).item_id.tolist()
        data = data[data.item_id.isin(popularity[:take_n_popular])]
    

    print('== Ending prefilter info ==')
    print('shape: {}'.format(data.shape))
    n_users = data.user_id.nunique()
    n_items = data.item_id.nunique()
    sparsity = float(data.shape[0]) / float(n_users*n_items) * 100
    print('# users: {}'.format(n_users))
    print('# items: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    return data


# User new feature:
#   "purchase frequency by one month"
def merge_purchase_freq_by_month(user_features, data):
    data = data[data.user_id.isin(user_features.user_id.unique().tolist())]
    data_quantity = data.groupby(['user_id']).quantity.sum().reset_index()
    data_months=data.groupby('user_id').apply(lambda x: ((x.week_no.max() - x.week_no.min())//4)).reset_index().rename(columns={0: 'n_months'})
    
    data = data_quantity.merge(data_months, on='user_id')
    data.loc[data.n_months < 1] = 1
    data['quantity_per_month'] = data.quantity / data.n_months
    user_features = user_features.merge(data[['user_id', 'quantity_per_month']], on='user_id', how='right')
    user_features['quantity_per_month'] = user_features.quantity_per_month.astype('int64')
    return user_features

# make pivot from department and user_id 
# >> purchases number of each department
def merge_pivot_by_departments(user_features, item_features, data):
    dt_tmp = user_features[['user_id']].merge(
        data.groupby(['user_id', 'item_id']).sales_value.sum().reset_index(), on='user_id').merge(
            item_features[['item_id', 'department']], on='item_id'
        ).groupby(['user_id', 'department']).sales_value.mean().reset_index()
    
    dep_pivot = pd.pivot_table(
            dt_tmp, 
            index='user_id', columns=['department'], 
            values='sales_value',
            # aggfunc='count', 
            fill_value=0
        ).reset_index() 
    user_features = user_features.merge(dep_pivot, on='user_id', how='left')

    return user_features

# User-Item new features:
#     "price",
#     "mean price by department"

def merge_ratio_mean_price_by_department(item_features, data):
    data = data[data.item_id.isin(item_features.item_id.unique().tolist())]
    
    s = item_features[['item_id', 'department']].merge(data[['item_id', 'sales_value', 'quantity']], on='item_id', how='right')
    s['price'] = s.sales_value / np.maximum(s.quantity, 1)
    s_mean = s.groupby('department').price.mean().reset_index()
    s_mean.columns = ['department', 'mean_price_by_department']
    
    item_features = item_features.merge(s_mean, on='department', how='left')
    item_features = item_features.join(s['price'])

    return item_features

# item new features:
# quantity selling per week

def merge_quantity_selling_per_week(item_features, data):
    d = data.copy()
    d = d[d.item_id.isin(item_features.item_id.unique().tolist())]
    d_sales = d.groupby('item_id').sales_value.sum().reset_index()
    d_weeks = d.groupby('item_id').apply(
        lambda x: x.week_no.max() - x.week_no.min()
        ).reset_index().rename(columns={0:'n_weeks'})
    d_weeks.loc[d_weeks.n_weeks == 0] = 1
    d_sales_weeks = d_weeks.merge(d_sales, on='item_id')
    d_sales_weeks['sales_by_week'] = d_sales_weeks.sales_value / d_sales_weeks.n_weeks
   
    item_features = item_features.merge(d_sales_weeks[['item_id', 'sales_by_week']], on='item_id', how='left')
    return item_features.fillna(0)
 
#   "ratio of price to average price of the department"
# + "ratio of mean prices by department"
def merge_price_rel_by_department(item_features, data):
    item_features = merge_ratio_mean_price_by_department(item_features, data)
    item_features['price_rel_mean_by_department'] = item_features.price / item_features.mean_price_by_department        
    
    return item_features