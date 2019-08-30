train_file=$1
item_vec_file=$2
brand_train_file=$3
brand_vec_file=$4
cate_train_file=$5
cate_vec_file=$6
../word2vec_model/word2vec -train $train_file -output $item_vec_file -cbow 0 -size 128 -window 30 -negative 50 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 100 -min-count 1
../word2vec_model/word2vec -train $brand_train_file -output $brand_vec_file -cbow 0 -size 128 -window 30 -negative 50 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 100 -min-count 1
../word2vec_model/word2vec -train $cate_train_file -output $cate_vec_file -cbow 0 -size 128 -window 30 -negative 50 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 100 -min-count 1
