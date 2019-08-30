"""
author：qiqiang.zhong
date:2019.07.09
produce item sim
"""
import operator
import os
import numpy as np
import sys


def load_item_vec(input_file):
    """
    Args:
        input_file:item_vec file
    return:
        dict key:itemid value：list[num1,num2...]
    """
    if not os.path.exists(input_file):
        return {}
    fp = open(input_file)
    linenum = 0
    item_vec = {}
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
        item_vec[itemid] = np.array([float(ele) for ele in item[1:]])
        linenum += 1
        print(linenum)
    fp.close()
    return item_vec


def cal_sim(itemid, item_vec, output_file):
    """
    Args:
        itemid:fixed itemid to cal item sim
        item_vec:the embedding vector
        output_file:the file of score result
    """
    if (itemid not in item_vec):
        return
    score = {}
    topk = 10
    fix_itemid = item_vec[itemid]
    for tempid in item_vec:
        if (tempid == itemid):
            continue
        temp_vec = item_vec[tempid]
        denomiator = np.linalg.norm(fix_itemid, ord=2) * np.linalg.norm(temp_vec,
                                                                        ord=2)  # linalg.norm  求范数 linear+algebra
        if (denomiator == 0):
            score[tempid] = 0
        else:
            score[tempid] = round(np.dot(temp_vec, fix_itemid) / denomiator, 5)
    fw = open(output_file, "w+")
    out_str = itemid + "\t"
    temp_list = []
    for zuhe in sorted(score.items(), key=operator.itemgetter(1), reverse=True)[:topk]:
        temp_list.append(str(zuhe[0]) + '\t' + str(zuhe[1]))
    out_str += ';'.join(temp_list)
    fw.write(out_str)
    fw.close()


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("usage:xx.py inputfile outputfile")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        item_vec = load_item_vec(input_file)
        print(item_vec)
        print(len(item_vec))
        cal_sim('/sku/brownsfashion.com/13247076', item_vec, output_file)
