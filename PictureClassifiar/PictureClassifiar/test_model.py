from PIL import Image
import torch
import torchvision

# 参数定义部分

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_path = "/PictureClassifiar/val_img/Pneumonia/00000011_007.jpg"
model_path = "/PictureClassifiar/save_model/best_model.pth"
class_list = {0: "NoFinding", 1: "Pneumonia"}  # 用于后续预测的时候可以使用，用预测到的标签来直接获取相应的类别


# 图片的读取
image = Image.open(image_path)

image = image.convert('RGB')
print(image)

# 图像的变换
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()]
)


image = transform(image)
print(image.shape)

model = torch.load(model_path)
# print(model)

image = torch.reshape(image, (1, 3, 224, 224))
image = image.to(device)

model.eval()


# 定义输出函数
def get_output(image):
    with torch.no_grad():
        output = model(image)

    print(output)
    # print(type(output))
    print("OUTPUT : ", output.argmax(1).item())

    final_output = output.argmax(1).item()
    class_id = final_output
    print('-----------------------------')
    print('This Image Maybe : ', class_list[final_output])
    print('-----------------------------')
    class_name = class_list[final_output]

    return [class_id, class_name]


print(get_output(image))
class_id, class_name = get_output(image)
print(class_id, class_name)




