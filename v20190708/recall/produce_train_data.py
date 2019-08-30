"""
author:qiqiang.zhong
date:2019.07.08
produce train data for item2vec
"""
import os
import sys


def produce_train_data(input_file, output_file):
    """
    Args:
        input_file:the raw log
        output_file:the user behavior extracted from the raw log
    """
    if not os.path.exists(input_file):
        return
    record = {}
    fp = open(input_file)
    for line in fp:
        split_list = line.strip().split('\t')
        if (len(split_list)) < 5:
            continue
        userid, item_id, isclick = split_list[0], split_list[2], split_list[-1]
        if (isclick == "FALSE"):  # str can not cast to bool
            continue
        if (userid not in record):
            record[userid] = []
        record[userid].append(item_id)
    fp.close()
    fw = open(output_file, 'w+')
    for userid in record:
        fw.write(" ".join(record[userid]) + "\n")
    fw.close()


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("usage:python xx.py inputfile outputfile")
    else:
        inputfile = sys.argv[1]
        outputfile = sys.argv[2]
        produce_train_data(inputfile, outputfile)
