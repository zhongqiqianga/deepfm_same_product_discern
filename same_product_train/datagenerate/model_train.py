import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn_pandas import DataFrameMapper


def train_model(model_path):
    '''
    用xgboost进行训练，对训练数据未进行采样
    :return:
    '''
    # dataset=pd.read_csv('/Users/looker/project/xmodel/same_product_judge/data/five_col_training_data.csv')
    dataset=pd.read_csv('/Users/looker/project/xmodel/same_product_judge/data/five_col_bags_data.csv')
    X=dataset.ix[:,dataset.columns!='label'].values
    Y=dataset.ix[:,dataset.columns=='label'].values.flatten().astype(np.int32)
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=33)
    ### fit model for train data
    model = XGBClassifier(learning_rate=0.1,
                          silent=1,
                          n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=1,  # 随机选择样本建立决策树
                          colsample_btree=1,  # 随机选取特征建立决策树
                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                          random_state=27,  # 随机数
                          objective='binary:logitraw'
                          )
    model.fit(x_train,y_train,eval_set=[(x_test, y_test)],early_stopping_rounds=100)

    ### plot feature importance
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(model,
                    height=0.5,
                    ax=ax,
                    max_num_features=4)
    plt.show()
    ### make prediction for test data
    y_pred = model.predict(x_test)
    ### model evaluate
    # accuracy = roc_auc_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    #保存模型
    joblib.dump(model, model_path)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))
    print("预测：")
    print(model.predict([[14.588458061218299,1.0,0.7698003649711609,0.0]]))


def save_pmml_model(model_path):
    '''
    用xgboost进行训练，对训练数据未进行采样,保存pmml文件供java解析
    :return:
    '''
    dataset = pd.read_csv('/Users/looker/project/xmodel/same_product_judge/data/five_col_training_data.csv')
    # X = dataset.ix[:, dataset.columns != 'label'].values
    # Y = dataset.ix[:, dataset.columns == 'label'].values.flatten().astype(np.int32)
    X = dataset.ix[:, dataset.columns != 'label']
    Y = dataset.ix[:, dataset.columns == 'label'].values.ravel()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=33)
    ### fit model for train data
    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=1,  # 随机选择样本建立决策树
                          colsample_btree=1,  # 随机选取特征建立决策树
                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                          random_state=27,  # 随机数
                          objective='binary:logistic'
                          )
    # model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=10,
    #           verbose=True)
    # 保存为pmml格式模型
    mapper = DataFrameMapper(
        [
            (['f0'], None),
            (['f1'], None),
            (['f2'], None),
            (['f3'], None)
        ]
    )
    pipeline = PMMLPipeline([('mapper', mapper), ("classifier", model)])
    pipeline.fit(X,Y)
    sklearn2pmml(pipeline, model_path, with_repr=True)




def buil_oversam_model(model_path):
    '''
        用xgboost进行训练，对训练数据进行上采样
        :return:
    '''
    dataset = pd.read_csv('/Users/looker/project/xmodel/same_product_judge/data/five_col_training_data.csv')
    X = dataset.ix[:, dataset.columns != 'label']
    Y = dataset.ix[:, dataset.columns == 'label']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=33)
    oversampler=SMOTE(random_state=0)
    x_train, y_train = oversampler.fit_sample(x_train, y_train)
    print(type(x_train))
    print(x_train)

    ### fit model for train data
    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=1000,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=1,  # 随机选择样本建立决策树
                          colsample_btree=0.8,  # 随机选择特征建立决策树
                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                          random_state=27,  # 随机数
                          objective='binary:logitraw'
                          )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=10,
              verbose=True)

    ### plot feature importance
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_importance(model,
                    height=0.5,
                    ax=ax,
                    max_num_features=4)
    plt.show()
    ### make prediction for test data
    y_pred = model.predict(x_test)
    joblib.dump(model,model_path)
    ### model evaluate
    accuracy = roc_auc_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))
    print("预测：")
    print(model.predict([[14.588458061218299,1.0,0.7698003649711609,0.0]]))


def process(train_data_path):
    """
    对缺失值进行处理，采用均值覆盖
    :param train_data_path:
    :return:
    """
    data = pd.read_csv(train_data_path)
    print(data['brand_score'].value_counts())
    brand_sum=0
    cate_sum=0
    count_zero_cate = 0
    count_zero_brand = 0
    for index in range(data.shape[0]):
        if(data.iloc[index,4]!=2):
            brand_sum+=data.iloc[index,4]
            count_zero_brand+=1
        if(data.iloc[index,5]!=2):
            cate_sum+=data.iloc[index,5]
            count_zero_cate+=1
    brand_aver=brand_sum/(count_zero_brand)
    cate_aver=cate_sum/(count_zero_cate)
    data.loc[data["brand_score"]==2,'brand_score']=brand_aver
    data.loc[data["cate_score"]==2,'cate_score']=cate_aver
    print(data['cate_score'].value_counts())
    data.drop(['raw_id','simi_id'],axis=1,inplace=True)
    data.to_csv("/Users/looker/project/xmodel/same_product_judge/data/five_col_bags_data.csv",index=None)
    return data


def test(data_path,model_path):
    """
    对剩下的数据进行预测
    :return:
    """
    data = pd.read_csv(data_path)
    # data = pd.read_csv(data_path).iloc[2200:]
    print(data['cate_score'].value_counts())
    brand_sum = 0
    cate_sum = 0
    count_zero_cate=0
    count_zero_brand=0
    for index in range(data.shape[0]):
        if (data.iloc[index, 4] != 2):
            brand_sum += data.iloc[index, 4]
            count_zero_brand+=1
        if (data.iloc[index, 5] != 2):
            cate_sum += data.iloc[index, 5]
            count_zero_cate+=1
    brand_aver = brand_sum / (count_zero_brand)
    cate_aver = cate_sum / (count_zero_cate)
    data.loc[data["brand_score"] == 2, 'brand_score'] = brand_aver
    data.loc[data["cate_score"] == 2, 'cate_score'] = cate_aver
    data_id=data.ix[:,[1,2]]
    data.drop(['raw_id', 'simi_id','label'], axis=1, inplace=True)
    model=joblib.load(model_path)
    X = data.ix[:, data.columns != 'label'].values
    y_pred = model.predict(X)
    data=data.reset_index(drop=True)
    print(y_pred)
    result=pd.DataFrame({'label':y_pred})
    data=pd.concat([result,data_id,data],axis=1)
    print(data.head(10))
    data.to_csv("/Users/looker/project/xmodel/same_product_judge/data/result_bags_test.csv", index=None)
    return data





if __name__=="__main__":
    train_data_path='/Users/looker/project/xmodel/same_product_judge/data/final_bags_test.csv'
    save_model_path="/Users/looker/project/xmodel/same_product_judge/data/model_bags.pkl"
    train_model(save_model_path)
