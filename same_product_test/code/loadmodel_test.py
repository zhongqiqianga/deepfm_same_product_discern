# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.externals import joblib
import sys

"""
加载模型，并对经过数据预处理的数据进行测试
"""


def test(data_path, model_path, result_path):
    """
    对数据进行预测
    :return:
    """
    data = pd.read_csv(data_path)
    print(data['cate_score'].value_counts())
    brand_sum = 0
    cate_sum = 0
    name_sum=0
    count_nozero_cate = 0
    count_nozero_brand = 0
    count_nozero_name = 0
    for index in range(data.shape[0]):
        if (data.iloc[index, 4] != 2):
            brand_sum += data.iloc[index, 4]
            count_nozero_brand += 1
        if (data.iloc[index, 5] != 2):
            cate_sum += data.iloc[index, 5]
            count_nozero_cate += 1
        if (data.iloc[index, 6] != 2):
            name_sum += data.iloc[index, 6]
            count_nozero_name += 1
    brand_aver = brand_sum / (count_nozero_brand)
    cate_aver = cate_sum / (count_nozero_cate)
    name_aver=name_sum/(count_nozero_name)
    data.loc[data["brand_score"] == 2, 'brand_score'] = brand_aver
    data.loc[data["cate_score"] == 2, 'cate_score'] = cate_aver
    data.loc[data["name_score"] == 2, 'name_score'] = name_aver
    data_id = data.ix[:, [1, 2]]
    data.drop(['raw_id', 'simi_id', 'label'], axis=1, inplace=True)
    model = joblib.load(model_path)
    X = data.ix[:, data.columns != 'label'].values
    y_pred = model.predict(X)
    data = data.reset_index(drop=True)
    result = pd.DataFrame({'label': y_pred})
    data = pd.concat([result, data_id, data], axis=1)
    data.to_csv(result_path, index=None)


if __name__ == "__main__":
    if (len(sys.argv) < 4):
        print("one of the file path not found ")
    else:
        data_path = sys.argv[1]
        model_path = sys.argv[2]
        result_path = sys.argv[3]
        test(data_path, model_path, result_path)
