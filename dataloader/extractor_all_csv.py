import pandas as pd
import glob
import os

def filter_and_save_to_csv(input_file_path, output_file_path):
    # 读取CSV文件
    df = pd.read_csv(input_file_path)

    # 打印列名以核实
    print(f"Columns in '{input_file_path}':", df.columns)

    required_columns = ['frame', ' confidence', ' gaze_0_x', ' gaze_0_y', ' gaze_1_x', ' gaze_1_y', ' pose_Tx', ' pose_Ty', ' pose_Rx', ' pose_Ry']

    try:
        # 尝试使用列名来选择列
        filtered_df = df[required_columns]
    except KeyError:
        print(f"Using index positions for '{input_file_path}'")
        # 获取列的索引位置
        column_indices = [df.columns.get_loc(col) for col in required_columns if col in df.columns]
        if len(column_indices) != len(required_columns):
            print(f"Missing columns in '{input_file_path}':", [col for col in required_columns if col not in df.columns])
            return
        # 使用列的索引位置来选择列
        filtered_df = df.iloc[:, column_indices]

    # 打印过滤后的数据的前几行
    print(f"Filtered data for '{input_file_path}':\n", filtered_df.head())

    # 将过滤后的DataFrame保存到新的CSV文件
    filtered_df.to_csv(output_file_path, index=False)
    print(f"Saved filtered data to '{output_file_path}'")

# 设置主文件夹路径
main_folder_path = r"EngageNet"  # 替换为“new”文件夹的路径
output_folder_path = r"EngageNet_csvs"  # 替换为输出CSV文件的文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 使用glob模块递归地获取所有子文件夹中的CSV文件
csv_files = glob.glob(os.path.join(main_folder_path, '**', '*.csv'), recursive=True)

# 遍历所有CSV文件并处理它们
for csv_file in csv_files:
    # 获取文件名
    file_name = os.path.basename(csv_file)
    # 设置输出CSV文件的路径
    output_file_path = os.path.join(output_folder_path, file_name)
    # 调用函数处理文件
    filter_and_save_to_csv(csv_file, output_file_path)