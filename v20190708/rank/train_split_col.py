"""
将值为列表的分为多个col
"""
import pandas as pd


def conversion_isdigit_list(str_list):
    lis = []
    str_list=str_list.replace('\'','').replace('[','').replace(']','').split(',')
    for i in str_list:
        i=i.strip()
        if i.isdigit():
            lis.append(int(i))
        else:
            pass
    return lis


def split_col_user(content):
    """
    将user.csv列表中有五个属性的值拆开
    """
    cols=['cate_30d','brand_30d','cate_7d','brand_7d','cate_1d','brand_1d']
    for col_index,col in enumerate(cols):
        dict_1={}
        dict_1[col+'_1'],dict_1[col+'_2'],dict_1[col+'_3'],dict_1[col+'_4'],dict_1[col+'_5']=[],[],[],[],[]
        for index in range(content.shape[0]):
            val=conversion_isdigit_list(content.iloc[index,col_index+1])
            print(val)
            dict_1[col+'_1'].append(val[0])
            dict_1[col + '_2'].append(val[1])
            dict_1[col + '_3'].append(val[2])
            dict_1[col + '_4'].append(val[3])
            dict_1[col + '_5'].append(val[4])
        dataset=pd.DataFrame(dict_1,columns=[col+'_1',col+'_2',col+'_3',col+'_4',col+'_5'])
        content=pd.concat([content,dataset],axis=1)
        print(col)
    return content

def split_col_product(content):
    """
    将name_word拆为三列
    """
    col='name_word'
    col_index=content.shape[1]-1
    dict_1 = {}
    dict_1[col + '_1'], dict_1[col + '_2'], dict_1[col + '_3'] = [], [], []
    for index in range(content.shape[0]):
        val = conversion_isdigit_list(content.iloc[index, col_index-1])
        dict_1[col + '_1'].append(val[0])
        dict_1[col + '_2'].append(val[1])
        dict_1[col + '_3'].append(val[2])
    dataset = pd.DataFrame(dict_1, columns=[col + '_1', col + '_2', col + '_3',])
    content = pd.concat([content, dataset], axis=1)
    # content=content.drop([col],axis=1)
    return content

def confix_data(content):
    """
    将所有数据打乱，并生成十万行的训练数据和两万行的测试数据
    """
    print(content.shape)
    content=content.sample(frac=1.0)
    content=content.reset_index(drop=True)
    train_lines=100000
    test_lines=20000
    content_train=content.iloc[:train_lines]
    contest_test=content.iloc[train_lines:train_lines+test_lines,]
    content_train.to_csv('/Users/looker/project/xmodel/v20190708/rank/example/data/train.csv',index=None)
    contest_test.to_csv('/Users/looker/project/xmodel/v20190708/rank/example/data/test.csv',index=None)





if __name__=='__main__':
    user_file_url='../data/ranking_data/user.csv'
    product_file_url='../data/ranking_data/product.csv'
    df=pd.read_csv(user_file_url)
    content=split_col_user(df)
    cols=['cate_30d','brand_30d','cate_7d','brand_7d','brand_1d','cate_1d']
    content=content.drop(cols,axis=1)
    content.to_csv('../data/ranking_data/split_user.csv',index=None)
    content_product=pd.read_csv(product_file_url)
    content_product=split_col_product(content_product)
    content_product.drop(['name_word'],axis=1,inplace=True)
    content_product.to_csv('../data/ranking_data/split_product.csv',index=None)
    # content_data=pd.read_csv('/Users/looker/project/xmodel/v20190708/rank/example/data/temp_train_data.csv')
    # confix_data(content_data)
