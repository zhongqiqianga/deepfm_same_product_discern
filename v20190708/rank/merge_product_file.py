import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pre_processing
from commons.hash import *



def read_rawproduct(filepath):
    """
    read the files 、 assign column names and return reasonable dataframe
    """
    names = ['pid', 'brand_name', 'cate2_7d', 'cate3_7d', 'site_id_7d', 'view_uv_7d', 'view_pv_7d', 'addcart_uv_7d',
             'addcart_pv_7d', 'pay_uv_7d', 'pay_pv_7d', 'price_7d', 'discount_7d']
    if (filepath == '../data/20190626/product_stat_7d.csv'):
        dataset = pd.read_csv(filepath)
        dataset.columns = names
    elif (filepath == '../data/20190626/product_stat_30d.csv'):
        names = ['pid', 'brand_name', 'cate2_30d', 'cate3_30d', 'site_id_30d', 'view_uv_30d', 'view_pv_30d',
                 'addcart_uv_30d',
                 'addcart_pv_30d', 'pay_uv_30d', 'pay_pv_30d', 'price_30d', 'discount_30d']
        dataset = pd.read_csv(filepath)
        dataset.columns = names
    else:
        dataset = pd.read_csv('../data/20190626/product_raw.csv',delimiter="\t", header=None)
        names = ['pid', 'name_word', 'other_one', 'other_two', 'other_three', 'other_four']
        dataset.columns = names
        dataset = dataset[['pid', 'name_word']]
    return dataset


def normalization(content, cols):
    for col in cols:
        content[col] = (content.col - content.col.min()) / (content.col.max() - content.col.min())
    return content


def standardization(content, cols):
    for col in cols:
        content.col = StandardScaler().fit_transform(content[col].values.reshape(-1, 1))
    return content





def merge_by_pid(content_one, content_two):
    new_content = pd.merge(content_one, content_two, on='pid')
    return new_content


def feature_encode(content, col, filepath):
    """
    feature encode with assigned file
    """
    content_index = pd.read_csv(filepath, sep='\t',header=None)
    mapping = {}
    for i in range(content_index.shape[0]):
        brand = content_index.iloc[i, 1]
        index = content_index.iloc[i, 0]
        mapping[brand] = int(index)
    content[col] = content[col].map(mapping)
    return content


def label_Encode(content, col, filename):
    """
        feature encode with sklean and output index file
    """
    label = pre_processing.LabelEncoder()
    labels = label.fit_transform(content[col].values.tolist())
    content[col] = pd.DataFrame({col: labels})
    with open(filename, 'w', encoding='UTF-8') as file_object:
        for index, item in enumerate(label.classes_):
            file_object.write(str(index) + "\t" + item.astype('str') + '\n')
    return content


def judge_siteid(site_id):
    """
    根据列表对site_id进行判断，返回if_site的值
    """
    if(site_id.startswith('pms-')):
        return 0
    else:
        return 1

def encode_name_word(content,col):
    """
    将name_word中的单词分割并利用字典记录下标,并将name_word打印出来
    """
    name_dict={}
    count=1
    for i in range(content.shape[0]):
        str_value=str(content_new.loc[i,col])
        print(str_value)
        ele_list=str_value.strip().split(',')
        for ele in ele_list:
            if(ele not in name_dict):
                name_dict[ele]=count
                count+=1
    return name_dict


def name2list(name_value, value_index: dict):
    """
    将json字符串按照index文件对应的dict进行筛选,排序后选择top5
    """
    name_value=str(name_value)
    if("nan"==name_value):
        return [0,0,0]
    value_list = []
    ele_list = name_value.strip().split(",")
    length=len(ele_list)
    if(length>=3):
        for ele in ele_list[-3:]:
            value_list.append(value_index[ele])
    else:
        for ele in ele_list:
            value_list.append(value_index[ele])
        for i in range(3 - length):
            value_list.append(0)
    return value_list





if __name__ == '__main__':
    filepath_7d_url = '../data/20190626/product_stat_7d.csv'
    filepath_30d_url = '../data/20190626/product_stat_30d.csv'
    filepath_raw_url = '../data/20190626/product_raw.csv'
    brand_index_url = "../data//20190626/brand_index.tsv"
    cate2_index_url = "../data/20190626/cate2_index.tsv"
    content_7d = read_rawproduct(filepath_7d_url)
    content_30d = read_rawproduct(filepath_30d_url)
    content_raw = read_rawproduct(filepath_raw_url)
    # merge three files and drop the same meaning columns
    content_new = merge_by_pid(content_7d, content_30d)
    content_new = merge_by_pid(content_new, content_raw)
    content_new = content_new.drop(columns=['brand_name_y'])
    content_new = feature_encode(content_new, 'brand_name_x', "../data/ranking_data/total_brand_index.txt")
    content_new = feature_encode(content_new, 'cate2_7d', "../data/ranking_data/cate2.txt")
    print(content_new.columns)
    content_new=content_new.drop(['cate2_30d','cate3_7d','cate3_30d'],axis=1)
    content_new['if_site'] = content_new['site_id_7d'].apply(lambda x: judge_siteid(x))
    content_new = content_new.drop(['site_id_7d', 'site_id_30d'], axis=1)
    # content_new['pid'] = content_new['pid'].apply(lambda x: md5_hash_trim(x))
    name_dict=encode_name_word(content_new,'name_word')
    print(name_dict)
    name_dict = sorted(name_dict.items(), key=lambda x: x[1])
    fw = open("../data/ranking_data/name_word_index.txt", 'w+')
    fw.write('\n'.join('{}\t{}'.format(x[1], x[0]) for x in name_dict))
    fw.close()
    content_new['name_word']=content_new['name_word'].apply(lambda x:name2list(x,name_dict))
    content_new.to_csv('../data/ranking_data/product.csv', index=False)

