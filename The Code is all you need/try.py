from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8*8*32, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # 适应二分类任务

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 8*8*32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
model = CNN()

checkpoint = torch.load("model")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


# 定义预处理的转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 指定目录
directory = r"M:\test"  # 替换为你的目录路径

# 读取目录下的文件名
file_names = os.listdir(directory)

# 加载并预处理图像
for file_name in file_names:
    # 检查文件后缀是否为图像格式
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        # 拼接图像路径
        image_path = os.path.join(directory, file_name)

        # 加载图像
        image = Image.open(image_path)


        # 图像进行预处理
        image_transformed = transform(image)
        image_transformed = image_transformed.unsqueeze(0)  # 增加一个维度，因为模型的输入是一个批次

        # 将输入数据放在相同的设备上
        image_transformed = image_transformed.to(device)

        # 执行推断
        with torch.no_grad():
            output = model(image_transformed)
            probabilities = torch.softmax(output, dim=1)
            positive_probability = probabilities[0, 1].item()  # 获取属于正样本的概率

        # 设置matplotlib图表大小
        plt.figure(figsize=(10, 10))

        # 仅展示前16张图片
        for i, file_name in enumerate(file_names[:8]):
            # 拼接图像路径
            image_path = os.path.join(directory, file_name)

            # 加载图像
            image = Image.open(image_path)

            # 图像进行预处理
            image_transformed = transform(image)
            image_transformed = image_transformed.unsqueeze(0)  # 增加一个维度，因为模型的输入是一个批次

            # 将输入数据放在相同的设备上
            image_transformed = image_transformed.to(device)

            # 执行推断
            with torch.no_grad():
                output = model(image_transformed)
                probabilities = torch.softmax(output, dim=1)
                positive_probability = probabilities[0, 1].item()  # 获取属于正样本的概率

            # 在4x4网格的第i+1个位置显示图像和概率
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(f'属于亚洲大黄蜂的概率: {positive_probability:.6f}')
            plt.axis('off')  # 不显示坐标轴

        plt.tight_layout()  # 自动调整子图间距
        plt.show()  # 显示图表