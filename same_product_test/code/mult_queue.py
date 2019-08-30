# -*- coding: UTF-8 -*-
from queue import Queue
import threading
import time
import requests
import json
import pandas as pd
import sys


def get_shoes_id(pid_womenshoes_url, pid_manshoes_url, pidList_url):
    """
    从接口中获取商品的所有ID，返回一个list
    :return:
    """
    temp_women_url = pid_womenshoes_url
    temp_man_url = pid_manshoes_url
    pid_list = []
    # 获取女鞋子pid
    a = 1
    while (a != 0):
        response_pid = requests.get(url=pid_womenshoes_url)
        pid_womenshoes_url = temp_women_url
        json_result = json.loads(str(response_pid.text))
        a = json_result['data']['total']
        if (a == 0):
            break
        pid_last = json_result['data']['product'][-1]['product_id'].strip()
        print(pid_last)
        for index, product in enumerate(json_result['data']['product']):
            pid_list.append(str(product['product_id'].strip()))
        print(len(pid_list))
        # if (len(pid_list) == 1000):
        #     return pid_list
        pid_womenshoes_url = pid_womenshoes_url + str(pid_last)
    # 获取男鞋pid
    b = 1
    while (b != 0):
        response_pid = requests.get(url=pid_manshoes_url)
        pid_manshoes_url = temp_man_url
        json_result = json.loads(str(response_pid.text))
        b = json_result['data']['total']
        if (b == 0):
            break
        pid_last = json_result['data']['product'][-1]['product_id'].strip()
        print(pid_last)
        for index, product in enumerate(json_result['data']['product']):
            pid_list.append(str(product['product_id'].strip()))
        print(len(pid_list))
        pid_manshoes_url = pid_manshoes_url + str(pid_last)
    print("log")
    fw = open(pidList_url, 'w+')
    fw.write('\n'.join(pid_list))
    fw.close()
    return pid_list


def generate_data_copy(pid, output_url):
    """
    根据id从接口中读取数据
    :param pid_list:
    :param output_url:
    :return:
    """
    prefix_raw_url = 'https://www.preferr.com/api/core/v1/products/'
    prefix_url = "https://www.preferr.com/api/core/v1/products/"
    latest_url = "/similar?page=pn:1;limit:20"
    num_candi = 20
    try:
        candicate_url = prefix_url + str(pid.strip()) + latest_url
        raw_url = prefix_raw_url + str(pid.strip())
        response_candi = requests.get(url=candicate_url)
        response_raw = requests.get(url=raw_url)
        result_candi = json.loads(str(response_candi.text))
        result_raw = json.loads(str(response_raw.text))
        raw_id = pid
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
        return data
    except BaseException:
        print("异常")


def saveFile(data):
    try:
        fileLock.acquire()
        data.to_csv(output_file_path, mode='a', header=False, index=False)
        fileLock.release()
    except BaseException:
        print("异常")
        fileLock.release()



exitFlag = 0


class myThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)
        process_data(self.name, self.q)
        print("Exiting " + self.name)


def process_data(threadName, q):
    while not exitFlag:
        if not workQueue.empty():
            data = q.get()
            content = generate_data_copy(data, output_file_path)
            # queueLock.acquire()
            saveFile(content)
            # queueLock.release()
            print("%s processing %s" % (threadName, data))
        # else:
            # queueLock.release()
        time.sleep(1)


if __name__ == "__main__":
    if (len(sys.argv) < 5):
        print("one of the file path or get_pid url not found")
    else:
        womenshoes_url = sys.argv[1]
        manshoes_url = sys.argv[2]
        output_file_path = sys.argv[3]
        pidList_url = sys.argv[4]
    fr=open(pidList_url,"r+")
    pid_list=[]
    for line in fr:
        pid_list.append(line.strip())
    print("pid_list长度是%s" % len(pid_list))
    print(pid_list)
    fileLock = threading.Lock()
    t1=time.time()
    # pid_list = get_shoes_id(womenshoes_url, manshoes_url,pidList_url)
    # pid_list = get_shoes_id(womenshoes_url, manshoes_url,pidList_url)
    print(len(pid_list))
    threadList = ["Thread-1", "Thread-2", "Thread-3", "Thread-4", "Thread-5", "Thread-6", "Thread-7", "Thread-8","Thread-9"]
    queueLock = threading.Lock()
    workQueue = Queue(400000)
    threads = []
    threadID = 1
    # 创建新线程
    for tName in threadList:
        thread = myThread(threadID, tName, workQueue)
        thread.start()
        threads.append(thread)
        threadID += 1
    # 填充队列
    queueLock.acquire()
    for pid in pid_list[0:500]:
        workQueue.put(pid)
    queueLock.release()
    # 等待队列清空
    while not workQueue.empty():
        pass
    # 通知线程是时候退出
    exitFlag = 1
    # 等待所有线程完成
    for t in threads:
        t.join()
    print("Exiting Main Thread")
    t2=time.time()
    print("耗费时间是:%s" % str(t2 - t1))

