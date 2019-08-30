python="/Users/looker/software/anaconda3/envs/tf/bin/python"
raw_log_file="../data/20190626/imprlog_m6.csv"
train_file="../data/20190626/ranking.txt"
item_vec_file="../data/20190626/item_vec_binary.txt"
item_sim_file="../data/20190626/sim_result.txt"
brand_train_file="../data/20190626/ranking_brand_del.txt"
brand_vec_file="../data/20190626/brand_vec_binary.txt"
cate_train_file="../data/20190626/ranking_cate_del.txt"
cate_vec_file="../data/20190626/cate_vec_binary.txt"


if [ -f $raw_log_file ];then
    $python produce_train_data.py $raw_log_file $train_file
    echo "train_data has been created"
else
    echo "no raw_log_file"
    exit
fi
if [ -f $train_file ];then
    sh train.sh $train_file $item_vec_file $brand_train_file $brand_vec_file $cate_train_file $cate_vec_file
    echo "item_vec_file has been created"
else
    echo "no train_file"
    exit
fi
if [ -f $item_vec_file ];then
    $python item_sim.py $item_vec_file $item_sim_file
    echo "item_sim_file hae been created"
else
    echo "no item_vec_file"
    exit
fi
