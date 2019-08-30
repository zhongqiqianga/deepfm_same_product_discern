
# set the path-to-files
TRAIN_FILE = "./data/train.csv"
TEST_FILE = "./data/test.csv"

SUB_DIR = "./output"


NUM_SPLITS = 3
RANDOM_SEED = 2017

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
'uid', 'pid',
    'cate_30d_1','cate_30d_2','cate_30d_3','cate_30d_4','cate_30d_5',
    'brand_30d_1', 'brand_30d_2', 'brand_30d_3', 'brand_30d_4', 'brand_30d_5',
    'cate_7d_1', 'cate_7d_2', 'cate_7d_3', 'cate_7d_4', 'cate_7d_5',
    'brand_7d_1', 'brand_7d_2', 'brand_7d_3', 'brand_7d_4', 'brand_7d_5',
    'cate_1d_1', 'cate_1d_2', 'cate_1d_3', 'cate_1d_4', 'cate_1d_5',
    'brand_1d_1', 'brand_1d_2', 'brand_1d_3', 'brand_1d_4', 'brand_1d_5',
    'if_click','brand_name_x','cate2_7d',
    'name_word_1','name_word_2','name_word_3','if_site'
]


NUMERIC_COLS = [
    'price_7d','discount_7d',
    'price_30d','discount_30d',
    'view_uv_7d', 'view_pv_7d', 'addcart_uv_7d', 'addcart_pv_7d', 'pay_uv_7d', 'pay_pv_7d',
    'view_uv_30d', 'view_pv_30d', 'addcart_uv_30d', 'addcart_pv_30d', 'pay_uv_30d', 'pay_pv_30d'
]

IGNORE_COLS = [

]
