import torch
from PIL import Image
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 用于后续预测的时候可以使用，用预测到的标签来直接获取相应的类别
class_list = {0: "NoFinding", 1: "Pneumonia"}
model_path = "save_model/best_model.pth"
model = torch.load(model_path)

model.to(device='cpu')


# 黑白照片（灰度图）识别
def isGrayMap(img, threshold=15):
    """
    入参：
    img：PIL读入的图像
    threshold：判断阈值，图片3个通道间差的方差均值小于阈值则判断为灰度图。
    阈值设置的越小，容忍出现彩色面积越小；设置的越大，那么就可以容忍出现一定面积的彩色，例如微博截图。
    如果阈值设置的过小，某些灰度图片会被漏检，这是因为某些黑白照片存在偏色，例如发黄的黑白老照片、
    噪声干扰导致灰度图不同通道间值出现偏差（理论上真正的灰度图是RGB三个通道的值完全相等或者只有一个通道，
    然而实际上各通道间像素值略微有偏差看起来仍是灰度图）
    出参：
    bool值
    """
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False


# 定义输出函数
def get_output(image):

    if isGrayMap(img=image) == True:
        # 灰度图转RGB
        image = image.convert("RGB")
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor()
        ])
        image = transform(image)
    else:
        # 正常图片变换
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor()
        ])
        image = transform(image)


    with torch.no_grad():
        # print(image.shape)
        image = torch.reshape(image, (1, 3, 224, 224))
        output = model(image)

    print(output)
    # print(type(output))
    print("OUTPUT : ", output.argmax(1).item())

    final_output = output.argmax(1).item()
    class_id = str(final_output)
    print('-----------------------------')
    print('This Image Maybe : ', class_list[final_output])
    print('-----------------------------')
    class_name = class_list[final_output]

    return [class_id, class_name]


if __name__ == '__main__':
    class_list = {0: "NoFinding", 1: "Pneumonia"}
    model_path = "save_model/best_model.pth"
    model = torch.load(model_path)
    model.to(device='cpu')

    image_path = "C:/Users/XMainL/PycharmProjects/DeepLearning/PictureClassifar/val_img/Pneumonia/00000003_007.jpg"
    img = Image.open(image_path)

    print(type(img))
    # print(img.shape)
    image = torch.reshape(img)
    get_output(image)

