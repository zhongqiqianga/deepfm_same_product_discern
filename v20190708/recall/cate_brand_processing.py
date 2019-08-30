"""
author:qiqiang.zhong
date:2019.07.11
process data before generating cate vec
"""

import os
import sklearn.preprocessing as pre_processing
import pandas as pd
import numpy as np
def cate_dict_gen(input_file):
    """
    根据product_stat_30d文件返回两个字典，其中brand_dict匹配PID和品牌，cate2_dict匹配PID和二级目录
    Arg:
        input_file: product_raw.csv with ItemID 、Brand and Cate2
    return：
        cate2_dict: Key:ItemIDs value: cate
        brand_dict: Key:ItemIDs Value: brand_name

    """
    if not os.path.exists(input_file):
        print("the file not found")
        return {}, {}
    fp = open(input_file)
    cate2_dict = {}
    brand_dict = {}
    for line in fp:
        ele_list = line.strip().split(",")
        if (len(ele_list) < 13):
            continue
        itemid, brand_name, cate2 = ele_list[0].replace('\"', ''), ele_list[1].replace('\"', ''), ele_list[2].replace(
            '\"', '')
        if (itemid not in cate2_dict):
            cate2_dict[itemid] = cate2
        else:
            continue
        if (itemid not in brand_dict):
            brand_dict[itemid] = brand_name
        else:
            continue
    return brand_dict, cate2_dict


def get_index_dict(brand_index_file, cate_index_file):
    """
    Function：
        根据品牌和目录索引文件返回两个字典，其中brand_index_dict匹配brand_name和index，cate_index_dict匹配cate2和index
    Args:
        brand_index_file: 品牌索引文件
        cate_index_file:二级目录索引文件
    return：
        brand_indx_dict： Key: brand_name Value:index
        cate2_index_dict: Key: cate2 Value:index
    """
    if (not os.path.exists(brand_index_file) or not os.path.exists(cate_index_file)):
        "brand_index file not found"
    fp = open(brand_index_file)
    brand_index_dict = {}
    for line in fp:
        ele_list = line.strip().split("\t")
        index, brand = ele_list[0], ele_list[1]
        if (brand not in brand_index_dict):
            brand_index_dict[brand] = int(index)
    fp.close()
    fp = open(cate_index_file)
    cate_index_dict = {}
    for line in fp:
        ele_list = line.strip().split("\t")
        index, cate = ele_list[0], ele_list[1]
        if (cate not in cate_index_dict):
            cate_index_dict[cate] = int(index)
    fp.close()
    return brand_index_dict, cate_index_dict


def merge_index_file(input_file, previous_dict, output_file):
    """
    function:
        将之前根据product_7d编码的brand.txt文件产生的dict和brand_index.tsv下标合并
    Args:
        input_file:brand
        previous_dict: the dict including brand and index created from bran
        out_file:the file path to be written
    """
    if (not os.path.exists(input_file)):
        print("brand.txt not found")
    brand_dict = {}
    start_index = 20000
    fp = open(input_file)
    for line in fp:
        ele_list = line.strip().split('  ')
        index, brand = ele_list[0], ele_list[1]
        if (brand not in brand_dict):
            brand_dict[brand] = index
    for key in brand_dict:
        if (key not in previous_dict):
            previous_dict[key] = start_index
            start_index += 1
    fp.close()
    previous_dict = sorted(previous_dict.items(), key=lambda x: x[1])
    # previous_dict=dict(previous_dict)
    print(previous_dict)
    fw = open(output_file, 'w+')
    fw.write('\n'.join('{}\t{}'.format(x[1], x[0]) for x in previous_dict))
    fw.close()
    print(len(previous_dict))


def label_Encode(input_file, filename):
    """
    Function：
        对文件product_raw.csv的brand进行编码，并输出编码后的类别
        feature encode with sklean and output index file

    """
    if (not os.path.exists(input_file)):
        print("file to be encoded not found")
    names = ['PID', 'other_one', 'brand_name', 'cate2', 'price', 'discount']
    content = pd.read_csv(input_file, encoding='utf-8', header=None, sep='\t', names=names)
    label = pre_processing.LabelEncoder()
    labels = label.fit_transform(content['brand_name'].values.tolist())
    content['brand_name'] = pd.DataFrame({'brand_name': labels})
    with open(filename, 'w', encoding='UTF-8') as file_object:
        for index, item in enumerate(label.classes_):
            file_object.write(str(index) + "\t" + item.astype('str').lower() + '\n')
    return content


def produce_brand_data(brand_dict, brand_index_dict, input_file, output_file):
    """
    Function：
        生成新的品牌训练文件
    Args:
        brand_dict:Key:ItemID Value:brand_name
        input_file: ranking.txt consists of ItemIDs
        output_file: the ItemID in ranking.txt has been replaced by brand_index
    """
    if (not os.path.exists(input_file)):
        print("input_file not found")
    fp = open(input_file)
    fw = open(output_file, "w+")
    for line in fp:
        itemID_list = line.strip().split(" ")
        for index, ele in enumerate(itemID_list):
            if (ele in brand_dict):
                itemID_list[index] = brand_dict[ele]
                if (brand_dict[ele] in brand_index_dict):
                    itemID_list[index] = brand_index_dict[itemID_list[index]]
        fw.write(" ".join(str(x) for x in itemID_list) + "\n")
    fw.close()
    fp.close()




def produce_cate_data(cate_dict, cate_index_dict, input_file, output_file):
    """
    Function：
        生成目录训练文件
    Args:
        cate_dict:Key:ItemID Value:cate2
        input_file: ranking.txt consists of ItemIDs
        output_file: the ItemID in ranking.txt has been replaced by cate2_index
    """
    if (not os.path.exists(input_file)):
        print("input_file not found")
    fp = open(input_file)
    fw = open(output_file, "w+")
    for line in fp:
        itemID_list = line.strip().split(" ")
        for index, ele in enumerate(itemID_list):
            if (ele in cate_dict):
                itemID_list[index] = cate_dict[ele]
                if (cate_dict[ele] in cate_index_dict):
                    itemID_list[index] = cate_index_dict[itemID_list[index]]
        fw.write(" ".join(str(x) for x in itemID_list) + "\n")
    fw.close()
    fp.close()



def delete_skr(input_file,output_file):
    """
    function:去除没有品牌对应的PID，可以在这里打印出剔除了哪些PID
    """
    fp=open(input_file)
    fw=open(output_file,'w+')
    epoch=5
    for line in fp:
        ele_list=line.strip().split(" ")
        for i in  range(epoch):
            for ele in ele_list:
                if(ele.isdigit()== False):
                    ele_list.remove(ele)
        fw.write(' '.join(ele_list)+"\n")
    fp.close()
    fw.close()

def get_vec_dict(itemId_vec):
    """
    Function:
        以字典形式返回每个值对应的向量
    Args:
        itemId_vec:item_vec file
        brand_vec:brand_vec file
        cate_vec: cate_vec file
    return:
        dict key:itemid or brand or cate value：list[num1,num2...]
    """
    fp = open(itemId_vec)
    linenum = 0
    vec_dict={}
    vec_dim = 128
    for line in fp:
        if (linenum == 0):
            linenum = linenum + 1
            continue
        item = line.strip().split(' ')
        if (len(item) < vec_dim + 1):
            continue
        itemid = item[0]
        if (itemid == "</s>"):
            continue
        vec_dict[itemid] = np.array([float(ele) for ele in item[1:]])
    fp.close()
    return vec_dict





if __name__ == "__main__":
    brand_dict, cate2_dict = cate_dict_gen("../data/20190626/product_stat_30d.csv")
    brand_index_dict, cate_index_dict = get_index_dict('../data/20190626/brand_index.tsv',
                                                       '../data/20190626/cate2_index.tsv')
    merge_index_file('../data/20190626/brand.txt', brand_index_dict, '../data/20190626/total_brand_index.txt')
    brand_index_dict, cate_index_dict = get_index_dict('../data/20190626/total_brand_index.txt',
                                                       '../data/20190626/cate2_index.tsv')
    produce_brand_data(brand_dict, brand_index_dict, "../data/20190626/ranking.txt", '../data/20190626/ranking_brand.txt')
    label_Encode('../data/20190626/product_raw.csv', '../data/20190626/brand_raw.txt')
    delete_skr('../data/20190626/ranking_brand.txt','../data/20190626/ranking_brand_del.txt')
    produce_cate_data(cate2_dict,cate_index_dict,'../data/20190626/ranking.txt','../data/20190626/ranking_cate.txt')
    delete_skr('../data/20190626/ranking_cate.txt','../data/20190626/ranking_cate_del.txt')
