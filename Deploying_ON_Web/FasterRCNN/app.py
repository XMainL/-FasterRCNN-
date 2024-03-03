""" Flask Deploy PyTorch Models"""
# 引入所需的包
import os
from flask import Flask, render_template, request, redirect, jsonify, Response, send_file
import io
from PIL import Image
import uuid
import torch
import torchvision
import predict_output
import json
from util import base64_to_pil

# 简单的的 Flask服务器
app = Flask(__name__)

# 参数列表
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        # 读取上传的图像
        image = base64_to_pil(request.json)
        print('Image Type : ', type(image))

        output = predict_output.get_output(image)
        # 测试点
        print(type(output))
        return output


@app.route("/upload", methods=["POST"], endpoint="upload")
def upload():
    """文件上传功能"""
    file_obj = request.files.get("file")
    if file_obj:
        # 获取文件的名字，包括后缀
        file_name = file_obj.filename
        with open(file_name, 'wb') as f:
            for line in file_obj:
                f.write(line)
        return "success !!"
    else:
        return "faild !!"


@app.route("/delete")
def delete_files():
    os.remove("C:/Users/NilEra/PycharmProjects/DeepLearning/faster-rcnn-pytorch-master/static/saved_imgs/img.jpg")


if __name__ == '__main__':
    # 地址需要是0.0.0.0，保证外机访问
    app.run(host="0.0.0.0", debug=True, port=5000)
