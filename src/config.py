import os


MODELS_PATH = os.path.join(os.pardir, 'models/')
DATA_PATH = os.path.join('data/')
TRAIN_SAMPLE_PATH = os.path.join(DATA_PATH, 'retail_train_sample.csv')
TEST_SAMPLE_PATH = os.path.join(DATA_PATH,'retail_test1.csv')
ITEM_FEATURES_PATH = os.path.join(DATA_PATH,'product.csv')
USER_FEATURES_PATH = os.path.join(DATA_PATH,'hh_demographic.csv')
ADDITIONAL_FUNCTIONS_PATH = 'src/'
ITEM_COL = 'item_id'
USER_COL = 'user_id'
ACTUAL_COL = 'actual'

# N neighbours
# 1st level recommendation
N_CANDIDATES = 50
# 2nd level 
N_PREDICT = 50
TARGET_COL = 'target'


VAL_MATCHER_WEEKS = 8
VAL_RANKER_WEEKS = 3