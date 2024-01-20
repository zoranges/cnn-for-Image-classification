import requests
from bs4 import BeautifulSoup
import os

# 创建保存图片的目录
save_dir = r"M:\mydataset\N"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 设置请求头，有些网站可能会检查这个来防止爬虫
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# 指定爬取的网站，这里以pixabay为例
url = 'https://www.duitang.com/search/?kw=%E6%87%92%E6%B4%8B%E6%B4%8B&type=feed'

# 发送GET请求
response = requests.get(url, headers=headers)

# 检查请求是否成功
if response.status_code == 200:
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 选取所有图片标签
    images = soup.find_all('img')

    # 循环遍历所有图片标签
    for img in images:
        # 获取图片的链接
        img_url = img.get('src') or img.get('data-src')
        if not img_url:
            continue
        # 下载图片
        try:
            img_data = requests.get(img_url).content
            img_name = os.path.basename(img_url)
            img_path = os.path.join(save_dir, img_name)
            with open(img_path, 'wb') as f:
                f.write(img_data)
                print(f"下载图片：{img_path}")
        except Exception as e:
            print(f"无法下载图片 {img_url}: {e}")

else:
    print("无法获取网页")
