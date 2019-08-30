import os
import requests
import json
import pandas as pd


def read_shoes_id(file_url):
    """
    读取id文件，返回id列表
    :param file_url:
    :return:
    """
    if(not os.path.exists(file_url)):
        print("pid file not found")
    else:
        pid_list=[]
        fr=open(file_url)
        for ele in fr:
            if(ele not in pid_list):
                pid_list.append(ele.strip())
            else:
                continue
    return pid_list


def generate_data(pid_list,output_url):
    """
    根据id从接口中读取数据
    :param pid_list:
    :param output_url:
    :return:
    """
    prefix_raw_url='https://www.preferr.com/api/core/v1/products/'
    prefix_url="https://www.preferr.com/api/core/v1/products/"
    latest_url="/similar?page=pn:1;limit:20"
    num_candi=20
    for index,pid in enumerate(pid_list):
        candicate_url = prefix_url + str(pid.strip()) + latest_url
        raw_url = prefix_raw_url + str(pid.strip())
        response_candi = requests.get(url=candicate_url)
        response_raw = requests.get(url=raw_url)
        result_candi = json.loads(str(response_candi.text))
        result_raw = json.loads(str(response_raw.text))
        raw_id = pid
        print(result_candi)
        print(result_raw)
        if (result_candi['data'] is None or result_raw['data'] is None):
            continue
        raw_name = result_raw['data']['name']
        raw_brand = result_raw['data']['brand']['name']
        raw_cate = result_raw['data']['category']['name']
        raw_id_list = [raw_id.strip()] * num_candi
        raw_name_list = [raw_name] * num_candi
        raw_brand_list = [raw_brand] * num_candi
        raw_cate_list = [raw_cate] * num_candi
        simi_name_list = []
        score_list = []
        simi_brand_list = []
        simi_id_list = []
        simi_cate_list = []
        for product in result_candi['data']['product']:
            simi_name_list.append(product['name'])
            simi_brand_list.append(product['brand']['name'])
            simi_id_list.append(product['product_id'].strip())
            simi_cate_list.append(product['category']['name'])
            score_list.append(float(product['score']))
        data_dict = {'raw_id': raw_id_list, 'simi_id': simi_id_list, 'raw_name': raw_name_list,
                     'raw_brand': raw_brand_list, 'raw_cate': raw_cate_list, 'simi_name': simi_name_list,
                     'simi_brand': simi_brand_list, 'simi_cate': simi_cate_list, 'simi_score': score_list,
                     'label': [0.1] * num_candi}
        data = pd.DataFrame(data_dict)
        order = ['label', 'raw_id', 'simi_id', 'raw_name', 'raw_brand', 'raw_cate', 'simi_name', 'simi_brand',
                 'simi_cate', 'simi_score']
        data = data[order]
        if (index == 0):
            data.to_csv(output_url, index=False)
        else:
            data.to_csv(output_url, mode='a', header=False, index=False)
        print(data)


def gen_transmit_data(output_url):
    """
    生成transmit的训练数据
    :return:
    """
    pid_list=['p-yoox-11518054UA','p-ff-12706798','p-ff-13576141','p-ff-14288191','p-nap-1137853']
    prefix_raw_url = 'https://www.preferr.com/api/core/v1/products/'
    prefix_url = "https://www.preferr.com/api/core/v1/products/"
    latest_url = "/similar?page=pn:1;limit:20"
    num_candi = 20
    for index, pid in enumerate(pid_list):
        candicate_url = prefix_url + str(pid.strip()) + latest_url
        raw_url = prefix_raw_url + str(pid.strip())
        response_candi = requests.get(url=candicate_url)
        response_raw = requests.get(url=raw_url)
        result_candi = json.loads(str(response_candi.text))
        result_raw = json.loads(str(response_raw.text))
        raw_id = pid
        print(result_candi)
        print(result_raw)
        if (result_candi['data'] is None or result_raw['data'] is None):
            continue
        raw_name = result_raw['data']['name']
        raw_brand = result_raw['data']['brand']['name']
        raw_cate = result_raw['data']['category']['name']
        raw_id_list = [raw_id.strip()] * num_candi
        raw_name_list = [raw_name] * num_candi
        raw_brand_list = [raw_brand] * num_candi
        raw_cate_list = [raw_cate] * num_candi
        simi_name_list = []
        score_list = []
        simi_brand_list = []
        simi_id_list = []
        simi_cate_list = []
        for product in result_candi['data']['product']:
            simi_name_list.append(product['name'])
            simi_brand_list.append(product['brand']['name'])
            simi_id_list.append(product['product_id'].strip())
            simi_cate_list.append(product['category']['name'])
            score_list.append(float(product['score']))
        data_dict = {'raw_id': raw_id_list, 'simi_id': simi_id_list, 'raw_name': raw_name_list,
                     'raw_brand': raw_brand_list, 'raw_cate': raw_cate_list, 'simi_name': simi_name_list,
                     'simi_brand': simi_brand_list, 'simi_cate': simi_cate_list, 'simi_score': score_list,
                     'label': [0.1] * num_candi}
        data = pd.DataFrame(data_dict)
        order = ['label', 'raw_id', 'simi_id', 'raw_name', 'raw_brand', 'raw_cate', 'simi_name', 'simi_brand',
                 'simi_cate', 'simi_score']
        data = data[order]
        for simi_id in simi_id_list:
            simi_raw_url = prefix_raw_url + str(simi_id.strip())
            simi_candicate_url = prefix_url + str(simi_id.strip()) + latest_url
            response_candi_simi = requests.get(url=simi_candicate_url)
            response_raw_simi = requests.get(url=simi_raw_url)
            result_candi_simi = json.loads(str(response_candi_simi.text))
            result_raw_simi = json.loads(str(response_raw_simi.text))
            raw_id_tran = simi_id
            print(result_candi_simi)
            print(result_raw_simi)
            if (result_candi_simi['data'] is None or result_raw_simi['data'] is None):
                continue
            raw_name_tran = result_raw_simi['data']['name']
            raw_brand_tran = result_raw_simi['data']['brand']['name']
            raw_cate_tran = result_raw_simi['data']['category']['name']
            raw_id_list_tran = [raw_id_tran.strip()] * num_candi
            raw_name_list_tran = [raw_name_tran] * num_candi
            raw_brand_list_tran = [raw_brand_tran] * num_candi
            raw_cate_list_tran = [raw_cate_tran] * num_candi
            trans_name_list = []
            trans_score_list = []
            trans_brand_list = []
            trans_id_list = []
            trans_cate_list = []
            for product_simi in result_candi_simi['data']['product']:
                trans_id_list.append(product_simi['product_id'].strip())
                trans_name_list.append(product_simi['name'])
                trans_brand_list.append(product_simi['brand']['name'])
                trans_cate_list.append(product_simi['category']['name'])
                trans_score_list.append(float(product_simi['score']))
            data_dict = {'raw_id': raw_id_list_tran, 'simi_id': trans_id_list, 'raw_name': raw_name_list_tran,
                         'raw_brand': raw_brand_list_tran, 'raw_cate': raw_cate_list_tran, 'simi_name': trans_name_list,
                         'simi_brand': trans_brand_list, 'simi_cate': trans_cate_list, 'simi_score': trans_score_list,
                         'label': [0.1] * num_candi}
            temp_data = pd.DataFrame(data_dict)
            order = ['label', 'raw_id', 'simi_id', 'raw_name', 'raw_brand', 'raw_cate', 'simi_name', 'simi_brand',
                     'simi_cate', 'simi_score']
            temp_data=temp_data[order]
            data=pd.concat([data,temp_data])
        if (index == 0):
            data.to_csv(output_url, index=False)
        else:
            data.to_csv(output_url, mode='a', header=False, index=False)
        print(data)


def gen_bag_data():
    pid_url='/Users/looker/project/xmodel/same_product_judge/data/bags_id'
    output_url='/Users/looker/project/xmodel/same_product_judge/data/bags_test_data.csv'
    pid_list = read_shoes_id(pid_url)
    generate_data(pid_list,output_url)



if __name__=="__main__":
    pid_url='/Users/looker/project/xmodel/same_product_judge/data/women_clothing_pid'
    output_url='/Users/looker/project/xmodel/same_product_judge/data/cloth_training_data.csv'
    pid_list=read_shoes_id(pid_url)[0:100]
    print(pid_list)
    generate_data(pid_list,output_url)
    # output_tran_url='/Users/looker/project/xmodel/same_product_judge/data/transmit_train_data.csv'
    # gen_transmit_data(output_tran_url)
    # gen_bag_data()
