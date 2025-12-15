import pandas as pd
import numpy as np
import pickle
import os
import json
from transformers import AutoTokenizer, AutoModel
import torch

# -------------------------- 1. 配置路径（请根据你的实际路径调整） --------------------------
EMBEDDING_MODEL = '/roberta'  # 你的RoBERTa模型路径
# 确保这个CSV文件包含 'video_path' 和 'per_frame_analysis' 这两列
# TEXT_CSV_PATH = 'text_frames_emotiw.csv'
TEXT_CSV_PATH = 'text_frames_daisee.csv'
# 新的保存路径，以区分旧的pkl文件
SAVE_DICT_PATH = 'embedding_dict_timeseries_daisee_16.pkl'


# -------------------------- 2. 你的JSON转换函数 (已集成) --------------------------
def convert_json_to_sentence_list(json_string):
    """
    将包含帧分析的JSON字符串解析并转换为带有时间戳的句子列表。
    格式: "At X.XX seconds, the student's state is: [States]."
    """
    if not isinstance(json_string, str):
        return []
    try:
        list_of_frames = json.loads(json_string)
        sentence_list = []
        
        # 使用 enumerate 获取索引 i (即公式中的 n)
        for i, frame_data in enumerate(list_of_frames):
            # 计算时间戳: n * 10 / 16
            # 假设每个切片代表总时长10秒中的1/16
            timestamp = i * 10.0 / 16.0
            
            # 获取状态内容 (这里保留了你原本只用 'Emotional Change' 的逻辑)
            # 如果需要把之前注释掉的 Gaze, Head 等加回来，可以在这里取消注释并拼接字符串
            content = (
                f"{frame_data.get('Gaze Direction', 'N/A')}, "
                f"{frame_data.get('Eyelid State', 'N/A')}, "
                f"{frame_data.get('Head Posture', 'N/A')}, "
                f"{frame_data.get('Mouth Movement', 'N/A')}, "
                f"{frame_data.get('Emotional Change', 'N/A')}"
            )
            
            # 按照要求构建句子
            # 例如: "At 0.62 seconds, the student's state is: neutral."
            sentence = f"At {timestamp:.3f} seconds, the student's state is: {content}."
            
            sentence_list.append(sentence)
            
        return sentence_list
    except (json.JSONDecodeError, TypeError) as e:
        print(f"处理JSON字符串时出错: {e} | 原始字符串: '{json_string[:100]}...'")
        return []


# -------------------------- 3. 加载NLP模型 (无需改动) --------------------------
def load_nlp_model(model_name=EMBEDDING_MODEL):
    nlp_tokenizer = None
    nlp_model = None
    try:
        print(f"Loading NLP Embedding Model: {model_name}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nlp_tokenizer = AutoTokenizer.from_pretrained(model_name)
        nlp_model = AutoModel.from_pretrained(model_name).to(device)
        nlp_model.eval()
        print(f"NLP Embedding Model loaded successfully to {device}.")
        return nlp_tokenizer, nlp_model, device
    except Exception as e:
        print(f"Error loading NLP model {model_name}: {e}")
        exit(1)


# -------------------------- 4. 计算并生成时序嵌入字典 (已修改) --------------------------
def generate_embedding_dict():
    # 加载模型
    tokenizer, model, device = load_nlp_model()
    
    # 读取文本CSV
    text_csv_data = pd.read_csv(TEXT_CSV_PATH)
    
    embedding_dict = {}
    
    print("开始为每个视频生成时序文本嵌入...")
    # 遍历CSV中的每一行
    for index, row in text_csv_data.iterrows():
        video_path = row['video_path']
        
        json_string = row['per_frame_analysis'] 
        subject_id = os.path.basename(video_path)

        sentence_list = convert_json_to_sentence_list(json_string)

        if not sentence_list:
            print(f"警告: 已跳过 {subject_id}，因为未能生成句子列表。")
            continue

        encoded_input = tokenizer(
            sentence_list,
            padding=True,
            truncation=True,
            max_length=512,  # 对每帧生成的句子长度进行限制
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)

        time_series_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

        embedding_dict[subject_id] = time_series_embeddings
        
        # 打印进度
        if (index + 1) % 50 == 0:
            print(f"已处理 {index + 1}/{len(text_csv_data)} 个视频...")

    with open(SAVE_DICT_PATH, 'wb') as f:
        pickle.dump(embedding_dict, f)
        
    if embedding_dict:
        first_key = list(embedding_dict.keys())[-1]
        print(f"示例: '{first_key}' 的嵌入形状为: {embedding_dict[first_key].shape}")


if __name__ == "__main__":
    # 确保保存目录存在
    save_dir = os.path.dirname(SAVE_DICT_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 生成并保存字典

    generate_embedding_dict()
