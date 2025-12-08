import os
import subprocess
import sys
import shutil
from PIL import Image
import glob
import csv

# 假设 feature_extraction_exe_path 是你的特征提取程序的路径
feature_extraction_exe_path = r""

# 获取 dataset 文件夹下的所有子文件夹
dataset_root_path = 'D:\\datasets\\Test'
dataset = os.listdir(dataset_root_path)

def is_image_all_black(image_path):
    """检查图片是否全黑"""
    with Image.open(image_path) as img:
        img = img.convert('L')  # 转换为灰度图
        pixels = list(img.getdata())
        return all(pixel < 10 for pixel in pixels)  # 允许一定的灰度值，避免噪声

def delete_black_image_and_csv_row(image_path, csv_file_path):
    """删除全黑图片并更新CSV文件"""
    os.remove(image_path)  # 删除图片
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)  # 读取所有行到列表

    # 删除包含全黑图片序号的行
    for row_index, row in enumerate(rows):
        if os.path.basename(image_path) in row:
            del rows[row_index]
            print(f"Deleted row from CSV for image: {os.path.basename(image_path)}")

    # 保存修改后的CSV文件
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def check_csv_rows(csv_file_path, min_rows):
    """
    检查CSV文件的行数是否满足最小行数要求。

    :param csv_file_path: CSV文件的路径。
    :param min_rows: 最小行数要求。
    :return: 如果行数大于或等于最小行数要求，则返回True，否则返回False。
    """
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        row_count = sum(1 for row in reader)
        return row_count >= min_rows

for ttv in dataset:
    # 获取 users 文件夹下的所有文件
    users_files = os.listdir(os.path.join(dataset_root_path, ttv))
    for user_file in users_files:
        # 假设 extract 是你要处理的特定文件的后缀或名称
        video_path = os.path.join(dataset_root_path, ttv, user_file)
        # 定义输出目录的路径
        output_dir_path = os.path.join('', ttv, os.path.splitext(user_file)[0])

        # 检查CSV文件是否存在，并且行数是否满足要求
        csv_file_name = os.path.splitext(user_file)[0] + '.csv'
        csv_file_path = os.path.join('', csv_file_name)
        if not os.path.exists(csv_file_path) or not check_csv_rows(csv_file_path, 280):
            print(f"CSV文件不存在或行数少于280，跳过视频 {user_file}")
            continue
        print(output_dir_path)
        # 检查输出目录是否已经存在
        if os.path.exists(output_dir_path):
            print(f"目录 {output_dir_path} 已存在，跳过处理。")
            continue  # 跳过当前循环的剩余部分，继续下一个循环
        # 确保输出目录存在
        os.makedirs(output_dir_path, exist_ok=True)
        # 构建命令行参数
        command = [
            feature_extraction_exe_path,
            "-f", video_path,
            "-out_dir", output_dir_path
        ]

        # 运行命令
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print("命令执行超时")
        except Exception as e:
            print(f"执行命令时出错: {e}")

        # 指定要删除的文件扩展名
        extensions = ['.avi', '.txt', '.hog']
        # 遍历每种扩展名
        # for ext in extensions:
        #     # 获取文件夹内所有匹配扩展名的文件列表
        #     files = glob.glob(os.path.join(output_dir_path, f'*{ext}'))
        #     # 遍历所有匹配的文件并删除
        #     for file_path in files:
        #         try:
        #             os.remove(file_path)
        #         except Exception as e:
        #             print(f"无法删除文件: {file_path}，原因: {e}")

        # 获取 output_dir_path 的最后一级目录名
        last_dir_name = os.path.basename(output_dir_path)

        # 创建新的输出目录路径，用于存放对齐后的图像
        output_dir_path_aligned = os.path.join(output_dir_path, last_dir_name + '_aligned')
        image_files = glob.glob(os.path.join(output_dir_path_aligned, '*.bmp'))
        # 遍历所有 .bmp 文件
        for image_path in image_files:
            # 打开图像文件
            with Image.open(image_path) as img:
                # 构建新的 .jpg 文件名
                jpg_path = os.path.splitext(image_path)[0] + '.jpg'

                # 保存为 .jpg 格式
                img.save(jpg_path, 'JPEG')

                # 删除原始的 .bmp 文件
                # os.remove(image_path)

        # # 获取上一级目录的路径
        # parent_dir = os.path.dirname(output_dir_path_aligned)
        # print(output_dir_path_aligned)
        # # 遍历指定文件夹中的所有文件
        # for filename in os.listdir(output_dir_path_aligned):
        #     # 构建完整的文件路径
        #     file_path = os.path.join(output_dir_path_aligned, filename)
        #     # 检查文件是否是图片（这里以常见的图片文件扩展名为例）
        #     if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        #         # 移动文件到上一级目录
        #         shutil.move(file_path, parent_dir)
        #         # print(f'Moved {filename} to {parent_dir}')