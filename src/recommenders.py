from encodings import search_function
from re import search
import pandas as pd
import numpy as np
import sys, os
# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from src.utils import calc_precision, calc_recall
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
from src.utils import process_user_item_features, convert_categorical_to_category
from src import config as cfg

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby([cfg.USER_COL, cfg.ITEM_COL])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases[cfg.ITEM_COL] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby(cfg.ITEM_COL)['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases[cfg.ITEM_COL] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        self.fitted = True

    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index=cfg.USER_COL, columns=cfg.ITEM_COL,
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.002, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
            print('new user: %d\tusers count: %d' % (user_id, len(list(self.userid_to_id.values()))))

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)[0] 
        top_rec = recs[1]  # take second item as the first = item_id
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]
        recommendations = recommendations[:N]
        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        userid = self.userid_to_id[user]
        user_item_matrix = csr_matrix(self.user_item_matrix).tocsr()

        res = [self.id_to_itemid[rec] for rec in model.recommend(userid=userid,
                                    user_items=user_item_matrix[userid],
                                    N=N,
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[999999]],
                                    recalculate_user=True)[0]]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        import src.config as cfg
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases[cfg.USER_COL] == user].head(N)

        res = top_users_purchases[cfg.ITEM_COL].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec for rec in similar_users[0]]
        similar_users = similar_users[1:]   # удалим юзера из запроса

        for u in similar_users:
            res.extend(self.get_own_recommendations(self.id_to_userid[u], N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res



class SecondLevelRecommendation:
    """
    **kwargs : train_ranker: pd.DataFrame
               candidates_df: pd.DataFrame, has two cols: cfg.USER_COL and cfg.ITEM_COL
               df_matcher: pd.DataFrame, concat of all data from the first level recommendation
               item_features: pd.DataFrame
               user_features: pd.DataFrame

    """ 
    def __init__(
                self,  
                ranker_train,
                ranker_val,
                balanced=False,
                **kwargs
                ):
        self.balanced = balanced
        
        self.items, self.users = None, None
        self.ranker_train = self.merge_candidates_ranker_data(train_ranker=ranker_train, **kwargs)
        self.ranker_val = self.merge_candidates_ranker_data(train_ranker=ranker_val, **kwargs)
        self.X_train, self.X_val, self.y_train, self.y_val = self.split_dataset(self.ranker_train, self.ranker_val)

        self.fitted = False
    def fit(self, model):
        if isinstance(model, cb.CatBoostClassifier):
            self.model = model
        else:
            return 'Model should be instance of catboost.CatBoostClassifier'

        X, y = self.X_train, self.y_train
        self.model.fit(X, y,  
                       cat_features=self.categorical_cols, 
                       eval_set=(self.X_val, self.y_val))
        self.fitted=True
        return self

    def split_dataset(self, data_train, data_test):
        import src.config as cfg

        if self.balanced:
            positive = data_train[data_train.target==1]
            negative = data_train[data_train.target==0]
            pos_upsampled = resample(positive,
                                    replace=True, # sample with replacement
                                    n_samples=len(negative), # match number in majority class
                                    random_state=27)
            data_train = pd.concat([negative, pos_upsampled])

        X_train = data_train.drop(cfg.TARGET_COL, axis=1)
        y_train = data_train[cfg.TARGET_COL]
        X_val = data_test.drop(cfg.TARGET_COL, axis=1)
        y_val = data_test[cfg.TARGET_COL]

        
        return X_train, X_val, y_train, y_val
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate_model(self, X_test=None):
        """Returns metrics dict of recalls and precisions 
        for @N in some range: {model_name:{precision@N: float},
                                          {recall@N: float}}"""
        result_metrics = dict(catboost_clf={})
        if X_test is None:
            X_test = self.X_val
                    
        preds = self.predict_proba(X_test[self.model.feature_names_])
        df_ranker_predict = X_test.copy()
        df_ranker_predict['proba_item_purchase'] = preds[:,1]
        def rerank(user_id):
            return df_ranker_predict[df_ranker_predict[cfg.USER_COL]==user_id].sort_values('proba_item_purchase', ascending=False).head(5)[cfg.ITEM_COL].tolist()
        result_eval_ranker = X_test.groupby(cfg.USER_COL)[cfg.ITEM_COL].unique().reset_index()
        result_eval_ranker.columns=[cfg.USER_COL, cfg.ACTUAL_COL]
        result_eval_ranker['catboost_clf'] = result_eval_ranker[cfg.USER_COL].apply(lambda user_id: rerank(user_id))
        self.result_eval_ranker = result_eval_ranker
        

        for N in (1, 10, 100, 200):
            p = sorted(calc_precision(result_eval_ranker, N), key=lambda x: x[1],reverse=True)
            for rec in p:
                result_metrics[rec[0]][f'Precision@{N}'] = rec[1]
            r = sorted(calc_recall(result_eval_ranker, N), key=lambda x: x[1],reverse=True)
            for rec in r:
                result_metrics[rec[0]][f'Recall@{N}'] = rec[1] 
        return result_metrics
        
    def merge_candidates_ranker_data(self, 
                                     candidates_df,
                                     train_ranker,
                                     df_matcher=None, 
                                     user_features=None,
                                     item_features=None
                                     ) -> pd.DataFrame:
        # get the whole merged dataset to get train-test for second level model
        df_ranker_train = train_ranker[[cfg.USER_COL, cfg.ITEM_COL]].copy()
        df_ranker_train['target'] = 1 #only purchases

        # merge candidates
        df_ranker_train = candidates_df.merge(df_ranker_train, on=[cfg.USER_COL, cfg.ITEM_COL], how='left') \
            .drop_duplicates(subset=[cfg.USER_COL, cfg.ITEM_COL])
        df_ranker_train['target'].fillna(0, inplace=True)      

        if self.users is None and self.items is None:
            self.users, self.items = process_user_item_features(df_matcher, 
                                                                user_features,
                                                                item_features)
                                                                
        df_ranker_train = df_ranker_train.merge(self.users, how='left', on='user_id').merge(self.items, how='left', on='item_id')
        df_ranker_train = convert_categorical_to_category(df_ranker_train)
        self.categorical_cols = df_ranker_train.select_dtypes('category').columns.tolist()
        return df_ranker_train
        
    def catboost_gridsearch(self) -> dict:
        X: pd.DataFrame = self.ranker_train.drop('target',axis=1)
        y: pd.DataFrame = self.ranker_train['target']

        train_dataset = cb.Pool(X, y, 
                        cat_features=self.categorical_cols)                                                      
    
        model = cb.CatBoostClassifier(loss_function='Logloss',  
                                      eval_metric='Accuracy')

        grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5,],
        'iterations': [50, 150, 500]}

        search_result = model.grid_search(
            param_grid=grid, 
            X=train_dataset, 
            cv=3, 
            plot=True)
        
        self.best_params, cv_results = search_result['params'], search_result['cv_results']

        print('best params: {}'.format(self.best_params))

        return self.best_params, cv_results