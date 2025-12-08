import os
from PIL import Image

def is_image_all_black(image_path):
    """检查图片是否全黑"""
    with Image.open(image_path) as img:
        img = img.convert('L')  # 转换为灰度图
        pixels = list(img.getdata())
        return all(pixel < 10 for pixel in pixels)  # 阈值设置为小于10，因为0是全黑，但可能会有轻微的噪声

def delete_black_images(directory):
    """删除全黑图片"""
    # 遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                if is_image_all_black(file_path):
                    # 立即删除图片
                    os.remove(file_path)
                    print(f"Deleted image: {file_path}")

# 替换为你的目标文件夹路径
target_directory = "D:\\datasets\\DAiSEE_extractor\\DataSet"

delete_black_images(target_directory)