""" Flask Deploy PyTorch Models"""
# 引入所需的包
import os
from flask import Flask, render_template, request, redirect, jsonify
import io
from PIL import Image
import uuid
import torch
import torchvision
import predict_output_PictureClassificar
import predict_output_FasterRCNN
import json
from util import base64_to_pil

# 简单的的 Flask服务器
app = Flask(__name__)

# 参数列表
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('LoadingPage.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     return redirect(request.url)
        # file = request.files.get('file')
        # if not file:
        #     return
        # # Get the image from post request
        # # 读取上传的图像
        # img_bytes = file.read()
        # image = Image.open(io.BytesIO(img_bytes))

        img = base64_to_pil(request.json)
        # print(type(img))
        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # 图像进行变换
        image = img
        # print(image.shape)
        # image = torch.reshape(image, (1, 3, 224, 224))
        # image = image.to(device)

        # Make prediction
        # 获得分类结果
        class_id, class_name = predict_output_PictureClassificar.get_output(image)
        print(class_id, class_name)
        result = class_name
        return jsonify(result=result, class_id=class_id, class_name=class_name)
    return None


@app.route('/predict_', methods=['GET', 'POST'])
def predict_():
    if request.method == 'POST':
        # Get the image from post request
        # 读取上传的图像
        image = base64_to_pil(request.json)
        print('Image Type : ', type(image))

        output = predict_output_FasterRCNN.get_output(image)
        # 测试点
        print(type(output))
        return output


@app.route("/delete")
def delete_files():
    os.remove("C:/Users/XMainL/PycharmProjects/DeepLearning/Deploying_ON_Web/static/saved_imgs/img.jpg")


@app.route("/check_folder")
def check_folder():
    img_path = "../saved_imgs/img.jpg"
    choice = os.path.exists(img_path)
    while(1):
        if choice == "True":
            print("Find The File!\n")
            return "True"
        else:
            continue


if __name__ == '__main__':
    # 地址需要是0.0.0.0，保证外机访问
    app.run(host="0.0.0.0", debug=True, port=5000)
