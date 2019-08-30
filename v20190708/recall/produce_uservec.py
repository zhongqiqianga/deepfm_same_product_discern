"""
author：qiqiang.zhong
date:2019.07.16
produce user vector
"""
import os
import numpy as np
import operator


def compute_uservec(ranking_file, item_vec_dict, output_file):
    """
    Function:
        根据用户行为序列和商品vector生成user vector，其中user vector是采取平均值
    Args:
        ranking_file: 带有uid的点击行为序列
        output_file: 将用户的embendding写入文件
        item_vec_dict:含有物品向量的字典查找表
    """
    if (not os.path.exists(ranking_file)):
        print("ranking file not found")
    user_vec_dict = {}
    fp = open(ranking_file)
    for line in fp:
        ele_list = line.strip().split(" ")
        userID = ele_list[0]
        if (userID not in user_vec_dict):
            sum = np.zeros((384,), dtype='float')
            for ele in ele_list[1:]:
                if (ele in item_vec_dict):
                    sum += item_vec_dict[ele]
                else:
                    continue
            user_vec_dict[userID] = np.round(sum / (len(ele_list) - 1),5)
    fp.close()
    fw = open(output_file, 'w+')
    for ele in user_vec_dict:
        fw.write(str(ele) + '\t' + " ".join(str(number) for number in user_vec_dict[ele])+'\n')
    fw.close()
    return user_vec_dict

def load_item_vec(input_file):
    """
    Function：
        根据商品向量文件返回dict用来生成
    Args:
        input_file:item_vec file
    return:
        dict key:itemid value：list[num1,num2...]
    """
    if not os.path.exists(input_file):
        return {}
    fp = open(input_file)
    item_vec = {}
    vec_dim = 384
    for line in fp:
        item = line.strip().split(' ')
        if (len(item) < vec_dim + 1):
            continue
        itemid = item[0]
        item_vec[itemid] = np.array([float(ele) for ele in item[1:]])
    fp.close()
    return item_vec



def user_item_sim(userid, user_vec,item_vec, output_file):
    """
    Function:
        计算用户和商品之间的相关性，并返回topK个商品
    Args:
        userid:fixed userid to cal item sim
        user_vec：the embedding user vector
        item_vec:the embedding item vector
        output_file:the file of score result
    """
    if (userid not in user_vec):
        return
    score = {}
    topk = 10
    fix_userid = user_vec[userid]
    for tempid in item_vec:
        temp_vec = item_vec[tempid]
        denomiator = np.linalg.norm(fix_userid, ord=2) * np.linalg.norm(temp_vec,
                                                                        ord=2)  # linalg.norm  求范数 linear+algebra
        if (denomiator == 0):
            score[tempid] = 0
        else:
            score[tempid] = round(np.dot(temp_vec,fix_userid ) / denomiator, 5)
    fw = open(output_file, "w+")
    out_str = userid + "\t"
    temp_list = []
    print(len(score))
    for zuhe in sorted(score.items(), key=operator.itemgetter(1), reverse=True)[:topk]:
        temp_list.append(str(zuhe[0]) + '\t' + str(zuhe[1]))
    out_str += ';'.join(temp_list)
    fw.write(out_str)
    fw.close()
    # topk = 10
    # fw = open(output_file, "w+")
    # for userid in user_vec:
    #     score = {}
    #     fix_userid = user_vec[userid]
    #     for tempid in item_vec:
    #         print(222222222)
    #         temp_vec = item_vec[tempid]
    #         denomiator = np.linalg.norm(fix_userid, ord=2) * np.linalg.norm(temp_vec,
    #                                                                         ord=2)  # linalg.norm  求范数 linear+algebra
    #         if (denomiator == 0):
    #             score[tempid] = 0
    #         else:
    #             score[tempid] = round(np.dot(temp_vec,fix_userid ) / denomiator, 5)
    #     out_str = userid + "\t"
    #     temp_list = []
    #     print(len(score))
    #     for zuhe in sorted(score.items(), key=operator.itemgetter(1), reverse=True)[:topk]:
    #         temp_list.append(str(zuhe[0]) + '\t' + str(zuhe[1]))
    #     out_str += ';'.join(temp_list)
    #     fw.write(out_str+'\n')
    # fw.close()

if __name__=="__main__":
    item_vec_url="../data/20190626/item_total_vec.txt"
    user_vec_url="../data/20190626/user_total_vec.txt"
    ranking_uid_url="../data/20190626/train_data_userid.txt"
    user_item_simurl='../data/20190626/user_item_result.txt'
    item_vec=load_item_vec(item_vec_url)
    user_vec=compute_uservec(ranking_uid_url,item_vec,user_vec_url)
    print(len(user_vec))
    user_item_sim('012295d4-9fd5-462e-818e-76c162c913f0',user_vec,item_vec,user_item_simurl)

