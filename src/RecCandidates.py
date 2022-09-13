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

from src.recommenders import MainRecommender
import src.config as cfg
from src.utils import process_user_item_features
from src.metrics import precision_at_k, recall_at_k


class RecommenderCandidates:
    def __init__(self, train, val,  **kwargs):
        self.train = train
        self.eval_df = val.groupby(cfg.USER_COL)[cfg.ITEM_COL].unique().reset_index(name=cfg.ACTUAL_COL)
        self.N = cfg.N_CANDIDATES
        self.model = MainRecommender(train, **kwargs)
        
        self.matcher_users = self.eval_df[cfg.USER_COL]
        self.baseline_model = None
    #  @staticmethod
    def _get_recommendations(self, user: int, model: str, N: int):
        if model == 'als':
            return self.model.get_als_recommendations(user, N=N)
        elif model == 'own':
            return self.model.get_own_recommendations(user, N=N)
        elif model == 'similar_items':
            return self.model.get_similar_items_recommendation(user, N=N)
        elif model == 'similar_users':
            return self.model.get_similar_users_recommendation(user, N=N)

    def get_candidates(self, eval_df=None, unstack=True):
        # get candidates on result (val_mather)
        if not self.baseline_model:
            self.baseline_model = self.choose_model()
        if eval_df is None: 
            eval_df = self.eval_df
        self.candidates = eval_df[cfg.USER_COL].apply(lambda x: self._get_recommendations(x, self.baseline_model, N=self.N)).reset_index()
        self.candidates.columns = ['user_id', 'candidates']
        assert isinstance(self.candidates, pd.DataFrame)

        if unstack:
            df = self.candidates.copy()
            items_df =  df.apply(lambda x: pd.Series(x['candidates']), axis=1) \
                        .stack() \
                        .reset_index(level=1, drop=True)
            items_df.name = cfg.ITEM_COL
            self.candidates = df.drop('candidates', axis=1).join(items_df) 
        return self.candidates
        

    def _get_all_recommendations(self):
        self.eval_df['als'] = self.matcher_users.apply(lambda x: self._get_recommendations(x, 'als', self.N))
        self.eval_df['own'] = self.matcher_users.apply(lambda x: self._get_recommendations(x, 'own', self.N))
        self.eval_df['similar_items'] = self.matcher_users.apply(lambda x: self._get_recommendations(x, 'similar_items', self.N))
        self.eval_df['similar_users'] = self.matcher_users.apply(lambda x: self._get_recommendations(x, 'similar_users', self.N))

    def choose_model(self):
        # get candidates on matcher (train) dataset
        self._get_all_recommendations()
        self.eval_result = sorted(self.calc_recall(self.eval_df, 50), key=lambda x: x[1], reverse=True)
        model_name = self.eval_result[0][0]
        return model_name

    @staticmethod
    def calc_recall(df_data, top_k):
        for col_name in df_data.columns[2:]:
            yield col_name, df_data.apply(lambda row: recall_at_k(row[col_name], row[cfg.ACTUAL_COL], k=top_k), axis=1).mean()

    @staticmethod
    def calc_precision(df_data, top_k):
        for col_name in df_data.columns[2:]:
            yield col_name, df_data.apply(lambda row: precision_at_k(row[col_name], row[cfg.ACTUAL_COL], k=top_k), axis=1).mean()
