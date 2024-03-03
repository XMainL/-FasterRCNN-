import torch

# 用于后续预测的时候可以使用，用预测到的标签来直接获取相应的类别
class_list = {0: "NoFinding", 1: "Pneumonia"}
model_path = "save_model/best_model.pth"
model = torch.load(model_path)


# 定义输出函数
def get_output(image):
    with torch.no_grad():
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