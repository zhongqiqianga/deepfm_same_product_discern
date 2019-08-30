"""
author:qiqiang.zhong
date:2019.08.13
process data
"""

from gensim import corpora
from gensim.similarities import Similarity
from nltk import word_tokenize
import pandas as pd
import numpy as np

def cal_sent_sim(sent_one,sent_two):
    """
    给定两个短语，计算相似度
    :param sent_one:
    :param sent_two:
    :return:
    """
    sents = [sent_one, sent_two]
    texts = [[word for word in word_tokenize(sent)] for sent in sents]
    print(texts)
    # print(texts)
    #  语料库
    dictionary = corpora.Dictionary(texts)
    print(dictionary)
    # 利用doc2bow作为词袋模型
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(corpus)
    similarity = Similarity('-Similarity-index', corpus, num_features=len(dictionary))
    # 获取句子的相似度
    new_sensence = sent_one
    test_corpus_1 = dictionary.doc2bow(word_tokenize(new_sensence))
    cosine_sim = similarity[test_corpus_1][1]
    print(cosine_sim)
    return cosine_sim
    # print("利用gensim计算得到两个句子的相似度： %.4f。" % cosine_sim)


def gen_train_data(raw_data_path):
    """
    截取已经标注的训练数据并返回
    :param raw_data_path:
    :return:
    """
    content=pd.read_csv(raw_data_path)
    # train_data=content.iloc[:2200]
    return  content


def gen_simi_list(content):
    """
    根据原始dataframe的属性，分别计算name、brand、cate的相似度，并写入list
    :param content:
    :return:
    """
    name_list=[]
    brand_list=[]
    cate_list=[]
    for index in range(content.shape[0]):
        raw_name = content.iloc[index, 3]
        raw_brand = content.iloc[index, 4]
        raw_cate = content.iloc[index, 5]
        simi_name = content.iloc[index, 6]
        simi_brand = content.iloc[index, 7]
        simi_cate = content.iloc[index, 8]
        if(raw_name is np.nan or simi_name is np.nan):
            name_list.append(2)
        else:
            name_score=cal_sent_sim(raw_name,simi_name)
            name_list.append(name_score)
        if (raw_brand is np.nan or simi_brand is np.nan):
            brand_list.append(2)
        else:
            brand_score = cal_sent_sim(raw_brand, simi_brand)
            brand_list.append(brand_score)
        if (raw_cate is np.nan or simi_cate is np.nan):
            cate_list.append(2)
        else:
            cate_score = cal_sent_sim(raw_cate, simi_cate)
            cate_list.append(cate_score)
    return name_list,brand_list,cate_list



def mod_train_data(train_data,name_list,brand_list,cate_list,temp_output,outfile_path):
    """
    根据name、brand、cate的相似度生成dataframe，并和原始数据进行拼接
    :param train_data:
    :param name_list:
    :return:
    """
    data_dict={'name_score':name_list,'brand_score':brand_list,'cate_score':cate_list}
    score_frame=pd.DataFrame(data_dict)
    final_frame=pd.concat([train_data,score_frame],axis=1)
    final_frame.to_csv(temp_output,index=None)
    drop_list=['raw_name','raw_brand','raw_cate','simi_name','simi_brand','simi_cate']
    final_frame.drop(drop_list,axis=1,inplace=True)
    final_frame.to_csv(outfile_path,index=None)
    print(score_frame.shape)
    print(score_frame.head(100))




if __name__=="__main__":
    raw_train_path='/Users/looker/project/xmodel/same_product_judge/data/bags_test_data.csv'
    temp_output='/Users/looker/project/xmodel/same_product_judge/data/bags_output_test.csv'
    final_train_path='/Users/looker/project/xmodel/same_product_judge/data/final_bags_test.csv'
    # raw_train_path = '/Users/looker/project/xmodel/same_product_judge/data/transmit_train_data.csv'
    # temp_output = '/Users/looker/project/xmodel/same_product_judge/data/temp_output_tran.csv'
    # final_train_path = '/Users/looker/project/xmodel/same_product_judge/data/final_training_data_tran.csv'
    train_data=gen_train_data(raw_train_path)
    name_list,brand_list,cate_list=gen_simi_list(train_data)
    mod_train_data(train_data,name_list,brand_list,cate_list,temp_output,final_train_path)
    cal_sent_sim("Women > Bags > Cross Body Bags","Women > Bags > Clutch Bags")