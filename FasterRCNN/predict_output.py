# 引入所需要的包
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import frcnn
from frcnn import FRCNN

image_path = "C:/Users/XMainL/PycharmProjects/DeepLearning/faster-rcnn-pytorch-master/img/00000001_000.jpg"
model_path = "C:/Users/XMainL/PycharmProjects/DeepLearning/faster-rcnn-pytorch-master/logs/best_epoch_weights.pth"
model = frcnn.FRCNN()
# model = torch.load(model_path)


def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width


# --------------------------------------------------------- #
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# --------------------------------------------------------- #
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


#   检测图片
def detect_image(self, image, crop=False, count=False):
    #   计算输入图片的高和宽
    image_shape = np.array(np.shape(image)[0:2])
    #   计算resize后的图片的大小，resize后的图片短边为600
    input_shape = get_new_img_size(image_shape[0], image_shape[1])
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    image = cvtColor(image)
    #   给原图像进行resize，resize到短边为600的大小上
    image_data = resize_image(image, [input_shape[1], input_shape[0]])
    #   添加上batch_size维度
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()

        #   roi_cls_locs  建议框的调整参数
        #   roi_scores    建议框的种类得分
        #   rois          建议框的坐标
        roi_cls_locs, roi_scores, rois, _ = self.net(images)
        #   利用classifier的预测结果对建议框进行解码，获得预测框
        results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                                         nms_iou=self.nms_iou, confidence=self.confidence)
        #   如果没有检测出物体，返回原图
        if len(results[0]) <= 0:
            return image

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

    #   设置字体与边框厚度
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
    #   计数
    if count:
        print("top_label:", top_label)
        classes_nums = np.zeros([self.num_classes])
        for i in range(self.num_classes):
            num = np.sum(top_label == i)
            if num > 0:
                print(self.class_names[i], " : ", num)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)
    # ---------------------------------------------------------#
    #   是否进行目标的裁剪
    # ---------------------------------------------------------#
    if crop:
        for i, c in list(enumerate(top_label)):
            top, left, bottom, right = top_boxes[i]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            dir_save_path = "img_crop"
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            crop_image = image.crop([left, top, right, bottom])
            crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
            print("save crop_" + str(i) + ".png to " + dir_save_path)
    # ---------------------------------------------------------#
    #   图像绘制
    # ---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        predicted_class = self.class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        # print(label, top, left, bottom, right)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image


def get_output(image):
    crop = False        # crop                指定了是否在单张图片预测后对目标进行截取
    count = False       # count               指定了是否进行目标的计数
    # 如果想要进行检测完的图片的保存, 利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
    r_image = FRCNN().detect_image(image, crop=crop, count=count)
    r_image.show()
    r_image.save("C:/Users/XMainL/PycharmProjects/DeepLearning/faster-rcnn-pytorch-master/static/saved_imgs/img.jpg")
    return r_image


if __name__ == "__main__":
    img = input('Input image filename:')
    image = Image.open(img)
    output = get_output(image)
    print(output)

