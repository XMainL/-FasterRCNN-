import torch
import torch.onnx
from model import ResNet50
import os


def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cuda'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    # model = ResNet50.ResNet50()  # 导入模型
    # model.load_state_dict(torch.load(checkpoint))  # 初始化权重
    model_path = "/PictureClassifiar/save_model/best_model.pth"
    model = torch.load(model_path)
    model.eval()
    model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    checkpoint = 'C:/Users/XMainL/PycharmProjects/DeepLearning/PictureClassifiar/save_model/best_model.pth'
    onnx_path = './test_2.onnx'
    input = torch.randn(1, 3, 224, 224)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path, device=device)

