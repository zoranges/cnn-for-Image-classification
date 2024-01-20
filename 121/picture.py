import os
from PIL import Image

# 指定目录
directory = r"M:/train/N"  # 替换为你的目录路径

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
        # 打印图片模式
        print(f'Image mode: {image.mode}')

        # 图片通道数量
        if image.mode == 'RGB':
            print('Number of channels: 3')
        elif image.mode == 'RGBA':
            print('Number of channels: 4')
        elif image.mode == 'L':
            print('Number of channels: 1')
        else:
            print('Other mode, possibly CMYK or other format with different channel numbers')
        # 检查图像是否有Alpha通道
        if image.mode == 'RGBA':
            # 转换图像
            image = image.convert('RGB')
            print('Alpha channel removed, image now has 3 channels.')
        else:
            print('Image does not have 4 channels to remove.')
        image.save(image_path)


