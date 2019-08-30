"""
编写kmeans代码，存储的数据结构是
"""
#导入模块
from same_product_judge.datagenerate.english_simi import cal_sent_sim
import numpy as np
import matplotlib.pyplot as plt





def cal_dist(sample1,sample2):
    """
    根据传进来的样本进行计算distance
    :param sample1:
    :param sample2:
    :return:
    """
    sample1.resize(4,)#对矩阵或者向量改变形状
    sample2.resize(4,)
    name_score=cal_sent_sim(sample1[1],sample2[1])
    brand_score=cal_sent_sim(sample1[2],sample2[2])
    cate_score=cal_sent_sim(sample1[3],sample2[3])
    return 1/float(name_score+brand_score+cate_score)



def initCentroids(dataSet,k):
    """
    初始聚类中心选择
    :param dataSet:
    :param k:
    :return:
    """
    numSamples,dim = dataSet.shape
    centroids = np.zeros((k,dim),dtype=np.object)
    for i in range(k):
        index = int(np.random.uniform(0,numSamples))
        centroids[i,:] = dataSet[index,:]
    print(centroids)
    return centroids




def kmeanss(dataSet,k):
    """
    K-means聚类算法，迭代
    :param dataSet:
    :param k:
    :return:
    """
    numSamples = dataSet.shape[0]
    clusterAssement = np.mat(np.zeros((numSamples,2)))
    clusterChanged = True
    #  初始化聚类中心
    centroids = initCentroids(dataSet,k)
    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            # 找到哪个与哪个中心最近
            for j in range(k):
                distance =cal_dist(centroids[j,:],dataSet[i,:])
                if distance<minDist:
                    minDist = distance
                    minIndex = j
              # 更新簇
            if clusterAssement[i,0]!=minIndex:
                clusterChanged = True
            clusterAssement[i,:] = minIndex,minDist
         # 坐标均值更新簇中心
        for j in range(k):
            min = 100000.0#计算到其他点的距离
            for index_one in range(numSamples):
                sum=0
                for index_two in range(numSamples):
                    if(  clusterAssement[index_one,0]==j and clusterAssement[index_two,0]==j):
                        print(clusterAssement[index_one,0])
                        sum+=cal_dist(dataSet[index_one,:],dataSet[index_two,:])
                if(sum<min):
                    min=sum
                    index=index_one
            # pointsInCluster = dataSet[np.nonzero(clusterAssement[:0].A==j)[0]]
            centroids[j,:] = dataSet[index,:]
        print("中心点跟新过")
    print('Congratulations,cluster complete!')
    return centroids,clusterAssement



def showCluster(dataSet,k,centroids,clusterAssement):
    """
    聚类结果显示
    :param dataSet:
    :param k:
    :param centroids:
    :param clusterAssement:
    :return:
    """
    numSamples,dim = dataSet.shape
    mark = ['or','ob']
    if k>len(mark):
        print('Sorry!')
        return 1
    for i in range(numSamples):
        markIndex = int(clusterAssement[i,0])
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=2)
    plt.show()
