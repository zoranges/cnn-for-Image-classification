import os

# 设置文件夹路径
directory = r"M:\训练集\N"

# 列出要删除的文件名
files_to_delete = [
    "ATT1_DSCN9647.jpg",
    "ATT2_HoireRDBeekeeperReport.jpg",
    "ATT4_20190921_152954.jpg",
    "ATT5_20190918_230100.jpg",
    "ATT6_20190918_235343.jpg",
    "ATT7_20190918_214742.jpg",
    "ATT256_47C3AA1B-C9D4-46FA-9D30-39655ACE1604.jpg",
    "ATT369_Langley 16 May.jpg",
    "ATT422_image001.jpg",
    "ATT472_9246EB0A-0DDF-420C-979D-C631943D3BFD.jpg",
    "ATT2302_image0-2.jpg",
    "ATT3067_IMG_20200926_091543.jpg",
    "ATT3141_30SEP20 doorbell cam.png",
    "ATT3142_KSal 30SEP20.jpg",
]

# 遍历文件名列表，删除每个文件
for file_name in files_to_delete:
    # 创建完整的文件路径
    file_path = os.path.join(directory, file_name)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，删除它
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        # 如果文件不存在，则打印消息
        print(f"File does not exist: {file_path}")
