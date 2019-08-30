"""
author:qiqiang.zhong
date:2019.07.15
process data before generating cate vec
"""
from v20190708.recall.cate_brand_processing import *
from v20190708.recall.item_sim import *
import os


def concatenate(item_vec_dict, brand_vec_dict, cate_vec_dict, brand_dict, brand_index_dict, cate_dict, cate_index_dict):
    """
    Function:
        传入brand cate pid由word2vec训练生成的向量
    Args：
        item_vec_dict: Key:pid index Value:vec array
        brand_vec_dict: Key:brand index Value:vec array
        cate_vec_dict: Key: cate index Value:vec array
        brand_dict: Key:pid value:brand name
        brand_index_dict:
        cate_dict:
        cate_index_dict:
    return:
        Key:pid Value:concateated vec
    """
    # 合并pid和brand的向量
    for ele in list(item_vec_dict.keys()):
        if (ele in brand_dict):
            brand_name = brand_dict[ele]
            if (brand_name in brand_index_dict):
                brand_index = brand_index_dict[brand_name]
                brand_index = str(brand_index)
                item_vec_dict[ele] = np.hstack((item_vec_dict[ele], brand_vec_dict[brand_index]))
            else:
                item_vec_dict.pop(ele)
        else:
            item_vec_dict.pop(ele)
    # 再将pid与brand合并后的向量与cate进行合并
    for ele in list(item_vec_dict.keys()):
        if (ele in cate_dict):
            cate_name = cate_dict[ele]
            if (cate_name in cate_index_dict):
                cate_index = cate_index_dict[cate_name]
                item_vec_dict[ele] = np.hstack((item_vec_dict[ele], cate_vec_dict[cate_index]))
            else:
                item_vec_dict.pop(ele)
        else:
            item_vec_dict.pop(ele)
    print(len(item_vec_dict))
    return item_vec_dict


def item_vec_written(item_vec_dict, output_file):
    """
        item_vec_dict: Key:pid Value:list
        outputfile: the path of file to write  item vec dict
    """
    if (not os.path.exists(output_file)):
        print("output_file not found")
    fw = open(output_file, 'w+')
    for ele in item_vec_dict:
        fw.write(str(ele) + ' ' + " ".join(str(number) for number in item_vec_dict[ele]) + "\n")
    fw.close()


if __name__ == "__main__":
    pid_vec_url = "../data/20190626/item_vec.txt"
    brand_vec_url = "../data/20190626/brand_vec.txt"
    cate_vec_url = "../data/20190626/cate_vec.txt"
    brand_dict_url = "../data/20190626/product_stat_30d.csv"
    pid_vec_dict = load_item_vec(pid_vec_url)
    brand_vec_dict = load_item_vec(brand_vec_url)
    cate_vec_dict = load_item_vec(cate_vec_url)
    brand_dict, cate2_dict = cate_dict_gen(brand_dict_url)
    brand_index_dict, cate2_index_dict = get_index_dict('../data/20190626/brand_index.tsv',
                                                        '../data/20190626/cate2_index.tsv')

    item_vec = concatenate(pid_vec_dict, brand_vec_dict, cate_vec_dict, brand_dict, brand_index_dict, cate2_dict,
                           cate2_index_dict)
    cal_sim("/sku/24sevres.com/253VQ", item_vec, "../data/20190626/sim_result.txt")
    item_vec_written(item_vec, '../data/20190626/item_total_vec.txt')
