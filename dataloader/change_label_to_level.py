import pandas as pd

# 读取Excel文件
df = pd.read_csv('TrainLabels.csv')

df = df[~df['label'].str.contains('SNP\(Subject Not Present\)', na=False)]

# 定义替换规则
replace_dict = {
    'Highly-Engaged': 3,
    'Engaged': 2,
    'Barely-engaged': 1,
    'Not-Engaged': 0
}

# 替换标签
df['label'] = df['label'].map(replace_dict)

# 将label列的数据类型转换为整数
df['label'] = df['label'].astype(int)

# 保存替换后的Excel文件，覆盖原文件
df.to_csv('label.csv', index=False)