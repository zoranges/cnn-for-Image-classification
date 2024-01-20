import os
from PIL import Image

# 指定目录
directory = r"M:\333"  # 替换为你的目录路径

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

        # 进行图像预处理
        # TODO: 添加你的图像预处理代码

        # 打印图像路径
        print("加载图像:", image_path)
