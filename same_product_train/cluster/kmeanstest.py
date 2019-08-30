# %load kmeanstest.py

from same_product_judge.cluster import kmeans
import  numpy as np

#从文件加载数据集
dataSet=[]
fileIn = open('/Users/looker/project/xmodel/same_product_judge/data/kmeans.csv')
for line in fileIn.readlines():
    lineArr = line.strip().split(',')
    dataSet.append([str(lineArr[0]),str(lineArr[1]),str(lineArr[2]),str(lineArr[3])])

#调用k-means进行数据聚类
dataSet = np.mat(dataSet)

k = 2
centroids,clusterAssement = kmeans.kmeanss(dataSet,k)
print(centroids)
print(clusterAssement)

# #显示结果
# kmeans.showCluster(dataSet,k,centroids,clusterAssement)
