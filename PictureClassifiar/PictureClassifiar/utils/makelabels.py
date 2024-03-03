# -*- coding: utf-8 -*-
import os
import re

path = "/PictureClassifiar/dataset/train"


# 更改文件名
def Rename_file(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            for imgs in os.listdir(file_path):
                img = os.path.splitext(imgs)  # 后缀
                newname = img[0] + '%' + file + img[1]
                newname = img[0] + '%' + file + img[1]
                os.rename(file_path + "\\" + imgs, file_path + "\\" + newname)
    print(".....done.......")


# 构建标签文件
def imglist(path):
    img = []
    label = []
    for file in os.listdir(path):
        file_img = os.path.join(path, file)
        if os.path.isdir(file_img):
            img = img + os.listdir(file_img)
    num = len(img)
    for i in range(num):
        imgs = img[i]
        name1 = imgs.split(".")[-2]  # 获取后缀之前的元素
        name2 = name1.split('%')[-1]  # 获取标签
        label = label + [int(name2)]
    assert len(img) == len(label)

    print("train img:", len(img), "......train label", len(label))


if __name__ == '__main__':
    imglist(path)

    # Rename_file(path)