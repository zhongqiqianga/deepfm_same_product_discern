# -*- coding: utf-8 -*-
import pandas as pd
import sys
import json
# import redis


def gen_dict(result_url):
    """
    根据结果文件生成dict
    :param result_url:
    :return:
    """
    content = pd.read_csv(result_url)
    result_dict = {}
    for index in range(content.shape[0]):
        label = content.iloc[index, 0]
        target_id = content.iloc[index, 1]
        cand_id = content.iloc[index, 2]
        if (label == 1):
            if (target_id not in result_dict):
                result_dict[target_id] = []
                result_dict[target_id].append(cand_id)
            else:
                result_dict[target_id].append(cand_id)
    return result_dict


def get_cluster(result_dict):
    """
    对传进来的result_dict进行聚类
    :return:
    """
    color_dict = {}
    for key_res in result_dict:
        if (len(color_dict) == 0):
            color_dict[key_res] = result_dict[key_res]
        else:
            temp_result = result_dict[key_res].copy()
            temp_result.append(key_res)
            flag_color = []  # 遇到冲突问题的时候，根据长度，标记该染哪一个颜色
            for ele in temp_result:
                for key_color in color_dict:
                    if (ele in color_dict):
                        flag_color.append(ele)
                    elif (ele in color_dict[key_color]):
                        flag_color.append(key_color)
            if (len(flag_color) == 0):
                color_dict[key_res] = result_dict[key_res]
            else:
                temp_len = len(color_dict[flag_color[0]])
                mark = flag_color[0]
                for temp_color in flag_color:
                    if (temp_len < len(color_dict[temp_color])):
                        temp_len = len(color_dict[temp_color])
                        mark = temp_color
                color_dict[mark] += temp_result
                for flag in flag_color:
                    if (mark != flag and flag in color_dict):
                        color_dict[mark] += color_dict[flag]
                        color_dict[mark].append(flag)
                        color_dict.pop(flag)
                color_dict[mark] = list(set(color_dict[mark]))
    for col in color_dict:
        if (col in color_dict[col]):
            color_dict[col].remove(col)
    return color_dict

def save_color(color_dict,color_dict_url):
    """
    将染色的结果保存
    :return:
    """
    color_dict_json=json.dumps(color_dict)
    file=open(color_dict_url,'w')
    file.write(color_dict_json)
    file.close()


def read_color_dict(color_dict_url):
    """
    读取染色结果
    :param color_dict_url:
    :return:
    """
    file = open(color_dict_url, 'r')
    js = file.read()
    dic = json.loads(js)
    file.close()







def get_color(result_dict, col_dict, pid):
    """
    根据商品pid返回所在的簇
    :param col_dict:
    :param pid:
    :return:
    """
    if (pid in col_dict):
        result = col_dict[pid]
        for ele in result_dict[pid]:
            result.remove(ele)
            result.insert(0, ele)
        return result
    else:
        for col in col_dict:
            if (pid in col_dict[col]):
                result = col_dict[col]
                result.append(col)
                for ele in result_dict[pid]:
                    result.remove(ele)
                    result.insert(0, ele)
                result.remove(pid)
                return result


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("one of the file not found")
    else:
        result_url = sys.argv[1]
        color_dict_url=sys.argv[2]
        result_dict = gen_dict(result_url)
        color_dict = get_cluster(result_dict)
        save_color(color_dict,color_dict_url)
        # result = get_color(result_dict, color_dict, "p-nap-1137853")
