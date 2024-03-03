"""
FileName : Split_Dataset
Usage    : Split The Dataset In The Specifies Folder
Anthor   : XMainL
"""
import os
from operator import index
from shutil import copy
from shutil import copy2
from shutil import rmtree
import random


def Split_Dataset(sourcePath, targetPath, train_rate, test_rate, val_rate):
    """
    数据集分割函数
    :param sourcePath: 源文件夹路径
    :param targetPath: 目标文件夹路径
    :param train_rate: 训练集的比率
    :param test_rate:  测试集的比率
    :param val_rate:   验证集的比率
    :return: 返回生成的文件夹
    """
    print("---===START DATASET SPLIT===---")
    class_names = os.listdir(sourcePath)
    # print(class_names)    # 测试点
    split_names = ['train', 'test', 'val']
    # 创建文件夹
    for split_name in split_names:
        split_name_path = os.path.join(targetPath, split_name)
        if os.path.isdir(split_name_path):
            pass
        else:
            os.mkdir(split_name_path)

        for class_name in class_names:
            class_name_path = os.path.join(split_name_path, class_name)
            if os.path.isdir(class_name_path):
                pass
            else:
                os.mkdir(class_name_path)

    # 数据集的划分
    for class_name in class_names:
        current_class_data_path = os.path.join(sourcePath, class_name)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index = list(range(current_data_length))
        random.shuffle(current_data_index)
        # 测试点
        print('current_class_data_path :', current_class_data_path)
        print('current_all_data:', current_all_data)
        print('current_data_length:', current_data_length)
        print('current_data_index:', current_data_index)

        train_path = os.path.join(os.path.join(targetPath, 'train'), class_name)
        test_path = os.path.join(os.path.join(targetPath, 'test'), class_name)
        val_path = os.path.join(os.path.join(targetPath, 'val'), class_name)

        train_ended = current_data_length * train_rate
        test_ended = current_data_length * test_rate
        val_ended = current_data_length * val_rate
        # 测试点
        # print(train_ended)
        # print(test_ended)
        # print(val_ended)

        current_index = 0
        train_num = 0
        test_num = 0
        val_num = 0

        for idx in current_data_index:
            sourcePath = os.path.join(current_class_data_path, current_all_data[idx])
            # 测试点
            print('============测试点===========')
            print('index:', idx)
            print('current_class_data_path:', current_class_data_path)
            print('current_all_data:', current_all_data[idx])
            print('sourcePath:', sourcePath)
            print('-', current_index)
            print('============================\n\n')
            if current_index < train_ended:
                copy(sourcePath, train_path)
                train_num = train_num + 1
            elif (current_index > train_ended) and (current_index <= train_ended + test_ended):
                print('测试点1')
                copy(sourcePath, test_path)
                test_num = test_num + 1
            else:
                print('测试点2')
                copy(sourcePath, val_path)
                val_num = val_num + 1

            current_index = current_index + 1
            # print(sourcePath)

        print("---==={}===---".format(class_name))
        print("{}类按照训练集:测试集:验证集 = {}:{}:{}的比例划分完成,共{}张图片"
              .format(class_name, train_rate, test_rate, val_rate, current_data_length))
        print("其中:")
        print("训练集{}:{}张".format(train_path, train_num))
        print("测试集{}:{}张".format(test_path, test_num))
        print("验证集{}:{}张".format(val_path, val_num))


if __name__ == "__main__":
    sourcePath = "C:/Users/XMainL/Desktop/sourcePath"
    targetPath = "C:/Users/XMainL/Desktop/targetPath"
    train_rate = 0.8
    test_rate  = 0.1
    val_rate   = 0.1

    Split_Dataset(sourcePath, targetPath, train_rate, test_rate, val_rate)

