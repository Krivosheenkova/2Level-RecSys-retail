import pandas as pd
import sys, os

def current_execute_directory():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        print('working in jupyter')
        return globals()['_dh'][0]

current_directory = current_execute_directory()
additional_functions_path = os.path.join(current_directory, os.pardir)
sys.path.insert(0, additional_functions_path)

import src.config as cfg
from src.utils import prefilter_items


class PreprocessMatcherData:
    def __init__(self, 
                 data: pd.DataFrame, 
                 item_features: pd.DataFrame, 
                 user_features: pd.DataFrame,
                 **kwargs):

        self.data = data.drop('Unnamed: 0', axis=1, errors='ignore')
        
        self.item_features = item_features
        self.user_features = user_features

        self.item_features.columns = [col.lower() for col in item_features.columns]
        self.item_features.rename(columns={'product_id': cfg.ITEM_COL}, inplace=True)
        
        self.user_features.columns = [col.lower() for col in user_features.columns]
        self.user_features.rename(columns={'household_key': cfg.USER_COL}, inplace=True)

        self.train_matcher, self.val_matcher, self.train_ranker, self.val_ranker = \
            self.matcher_ranker_split(self.data)
        
        self.train_matcher = prefilter_items(data=self.train_matcher, 
                                             item_features=self.item_features,
                                             **kwargs)

        self.common_users = self.train_matcher[cfg.USER_COL].unique().tolist()
        self.make_warm_start(self.common_users)
        self.df_matcher = pd.concat([self.train_matcher, self.val_matcher])

    def matcher_ranker_split(self, data: pd.DataFrame, week_col='week_no'):

        train_matcher = data[data[week_col] < data[week_col].max() - (cfg.VAL_MATCHER_WEEKS + cfg.VAL_RANKER_WEEKS)]
        val_matcher = data[(data[week_col] >= data[week_col].max() - (cfg.VAL_MATCHER_WEEKS + cfg.VAL_RANKER_WEEKS)) &
                        (data[week_col] < data[week_col].max() - (cfg.VAL_RANKER_WEEKS))]

        train_ranker = val_matcher.copy()
        val_ranker = data[data[week_col] >= data[week_col].max() - cfg.VAL_RANKER_WEEKS]

        print('input data splitted:')
        print('train_matcher: {}-{} weeks'.format(train_matcher[week_col].min(), train_matcher[week_col].max()))
        print('val_matcher: {}-{} weeks'.format(val_matcher[week_col].min(), val_matcher[week_col].max()))
        print('train_ranker: {}-{} weeks'.format(train_ranker[week_col].min(), train_ranker[week_col].max()))
        print('val_ranker: {}-{} weeks'.format(val_ranker[week_col].min(), val_ranker[week_col].max()))

        return train_matcher, val_matcher, train_ranker, val_ranker

    def make_warm_start(self, train_users):
        self.val_matcher = self.val_matcher.loc[self.val_matcher[cfg.USER_COL].isin(train_users)]
        self.train_ranker = self.train_ranker.loc[self.train_ranker[cfg.USER_COL].isin(train_users)]
        self.val_ranker = self.val_ranker.loc[self.val_ranker[cfg.USER_COL].isin(train_users)]

    def get_join_matcher(self):
        # get joined set of data for the first level (matching)
        return pd.concat([self.train_matcher, self.val_matcher])
        
    @staticmethod
    def print_stats_data(data, name):
        print(name)
        print(f'shape: {data.shape}\titems: {data[cfg.ITEM_COL].nunique()}\tusers: {data[cfg.USER_COL].nunique()}')

    def print_all_data_stats(self):
        
        self.print_stats_data(self.train_matcher,'train_matcher')
        self.print_stats_data(self.val_matcher,'val_matcher')
        self.print_stats_data(self.train_ranker,'train_ranker')
        self.print_stats_data(self.val_ranker,'val_ranker')

