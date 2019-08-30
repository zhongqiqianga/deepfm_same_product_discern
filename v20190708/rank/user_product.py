import sys
import operator
from commons.hash import *
from v20190708.recall.cate_brand_processing import *
import json


def convert2list(s: str, value_index: dict):
    """
    将json字符串按照index文件对应的dict进行筛选,排序后选择top5
    """
    s = s.replace("\"\"", "\"")
    topk = 5
    if s == '':
        return [0, 0, 0, 0, 0]
    j = json.loads(s)
    d = {value_index[k]: v for k, v in j.items() if k in value_index}
    value_list = []
    length = len(d)
    if (length >= 5):
        for ele in sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:topk]:
            value_list.append(ele[0])
    else:
        for ele in sorted(d.items(), key=operator.itemgetter(1), reverse=True):
            value_list.append(ele[0])
        for i in range(5 - length):
            value_list.append(0)
    return value_list


def load_user_file(file_path):
    """
    读取用户特征文件，并对属性的值进行改变，返回dataframe
    """
    if (not os.path.exists(file_path)):
        print("file not found")
    user_dataset = pd.read_csv(file_path)
    names = ['uid', 'cate_30d', 'brand_30d', 'cate_7d', 'brand_7d', 'cate_1d', 'brand_1d', 'u_tag_view_1d']
    user_dataset.columns = names
    # 对uid进行hash
    user_dataset['uid'] = user_dataset['uid'].apply(md5_hash_trim)
    user_dataset = user_dataset.drop(['u_tag_view_1d'], axis=1)
    # 对值为json数据的其他属性进行列表转换
    for ele in names[1:-1]:
        if (ele.startswith('brand')):
            user_dataset[ele] = user_dataset[ele].apply(lambda x: convert2list(x, brand_index))
        else:
            user_dataset[ele] = user_dataset[ele].apply(lambda x: convert2list(x, cate_index))
    print(user_dataset.head(10))
    user_dataset.to_csv('../data/ranking_data/user.csv', index=False)
    return user_dataset


def judge_siteid(site_id):
    """
    根据列表对site_id进行判断，返回if_site的值
    """
    if (site_id.startswith('pms-')):
        return 0
    else:
        return 1




def merge_userclick_product(userclick_file, product_file, user_file, output_file):
    """
    合并用户历史点击的行为序列和商品特征文件
    """
    if (not os.path.exists(userclick_file) or not os.path.exists(product_file)):
        print("input file not found")
    product_data = pd.read_csv(product_file)
    print(product_data.shape)
    click_data = pd.read_csv(userclick_file)
    user_data = pd.read_csv(user_file)
    names = ['uid', 'date', 'pid', 'did', 'if_click']
    click_data.columns = names
    click_data = click_data.drop(columns=['date', 'did'])
    # click_data['pid']=click_data['pid'].apply(md5_hash_trim)
    # click_data['uid'] = click_data['uid'].apply(md5_hash_trim)
    # Showing ratio
    print("Percentage of not click transactions: ",
          len(click_data[click_data.if_click == 0]))  # 打印正样本数目
    print("Percentage of click transactions: ",
          len(click_data[click_data.if_click == 1]))  # 打印负样本数目

    # 对日志进行下采样
    X = click_data.ix[:, click_data.columns != 'if_click']  # 取出所有属性，不包含class的这一列
    y = click_data.ix[:, click_data.columns == 'if_click']  # 另y等于class这一列

    # Number of data points in the minority class
    number_click = len(click_data[click_data.Class == 1])  # 计算出class这一列一号元素有多少个
    click_indices = np.array(click_data[click_data.Class == 1].index)  # 取出class这一列所有等于1的行索引

    # Picking the indices of the normal classes
    not_click_indices = click_data[click_data.if_click == 0].index  # 取出class这一列所有等于0的行索引

    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_notclick_indices = np.random.choice(not_click_indices, 5 * number_click,
                                               replace=False)  # 随机选择和1这个属性样本个数相同的0样本
    random_notclick_indices = np.array(random_notclick_indices)  # 转换成numpy的格式

    # Appending the 2 indƒices
    under_sample_indices = np.concatenate([click_indices, random_notclick_indices])  # 将正负样本拼接在一起

    # Under sample dataset
    under_sample_data = click_data.iloc[under_sample_indices, :]  # 下采样数据集

    # Showing ratio
    print("Percentage of normal transactions: ",
          len(under_sample_data[under_sample_data.if_click == 0]) / len(under_sample_data))  # 打印正样本数目
    print("Percentage of fraud transactions: ",
          len(under_sample_data[under_sample_data.if_click == 1]) / len(under_sample_data))  # 打印负样本数目
    print("Total number of transactions in resampled data: ", len(under_sample_data))  # 打印总数量
    click_data.to_csv('user_click_file.csv')
    table = pd.read_csv('user_click_file', chunksize=1000000)
    count = 0
    for df in table:
        train_data = pd.merge(product_data, df, on='pid')
        train_data = pd.merge(user_data, train_data, on='uid')
        if (count == 0):
            train_data.to_csv(train_data.csv)
            count += 1
        else:
            train_data.to_csv(train_data.csv, mode='a', header=False)


if __name__ == "__main__":
    user_click_url = '../data/20190626/imprlog_m6.csv'
    product_file_url = '../data/ranking_data/product.csv'
    brand_index_url = "../data/ranking_data/total_brand_index.txt"
    cate_index_url = "../data/ranking_data/cate2.txt"
    user_file_url = "../data/ranking_data/u_feature.csv"
    user_mod_url = "../data/ranking_data/user.csv"
    train_data_url = "../data/ranking_data/train_data.csv"
    brand_index, cate_index = get_index_dict(brand_index_url, cate_index_url)
    load_user_file(user_file_url)
    # merge_userclick_product(user_click_url,product_file_url,user_mod_url,train_data_url)
