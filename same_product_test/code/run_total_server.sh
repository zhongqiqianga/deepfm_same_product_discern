python="/Users/looker/software/anaconda3/envs/tf/bin/python"
womenshoes_url="http://47.252.3.121:8012/api/search/v1/products/list?cate=women-shoes&limit=1000&productId="
manshoes_url="http://47.252.3.121:8012/api/search/v1/products/list?cate=men-shoes&limit=1000&productId="
raw_data_file='../data/raw_data_server.csv'
final_train_file='../data/modify_train_data.csv'
model_path='../model/model.pkl'
result_path='../data/test_result.csv'
color_dict_path='../data/color_dict.csv'
pidList_path='../data/pidList.txt'

$python mult_queue.py $womenshoes_url $manshoes_url $raw_data_file $pidList_path
#$python generate_raw_data.py $womenshoes_url $manshoes_url $raw_data_file
if [ -f $raw_data_file ];then
    $python modify_train_data.py $raw_data_file $final_train_file
    echo "modify_train_data has been created"
else
    echo "no raw_train_data"
    exit
fi
if [ -f $final_train_file ];then
    $python loadmodel_test.py $final_train_file $model_path $result_path
    echo "test_result has been created"
else
    echo "no final_train_data"
    exit
fi
if [ -f $result_path ];then
    $python colouration.py  $result_path  $color_dict_path
    echo "color_dict_data has been created"
else
    echo "no test_result"
    exit
fi