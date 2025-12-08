import os
import cv2
import torch
import json
from PIL import Image
import torch.backends.cudnn as cudnn
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
from dataloader.video_dataloader import train_data_loader, test_data_loader
import numpy as np
import warnings
import traceback
import base64
warnings.filterwarnings("ignore", category=UserWarning)
import requests
import time
import matplotlib.pyplot as plt
import shutil
import itertools
import datetime
import tqdm
import random
from sklearn.metrics import confusion_matrix

# -------------------------- 配置参数 --------------------------
# 模型与数据路径（对齐你的DAiSEE数据集）
TRAIN_LIST_FILE = 'DAiSEE_Test_set.txt'
TEST_LIST_FILE = 'DAiSEE_Test_set.txt'
SL_FILE = 'SL_DAiSEE.csv'
log_txt_path = '/log/' + 'DAiSEE-set-log.txt'

# TRAIN_LIST_FILE = 'EngageNet_Test_set.txt'
# TEST_LIST_FILE = 'EngageNet_Test_set.txt'
# SL_FILE = 'SL_EngageNet.csv'
# log_txt_path = '/log/' + 'EngageNet-set-log.txt'

random.seed(42)  
np.random.seed(42) 
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


# 视频处理参数（完全复用你Flask代码中的配置，避免不一致）
FRAME_SAMPLING_RATE = 16  # 每隔16帧采集1帧（与你Flask代码一致）
MAX_FRAMES = 256           # 最多处理16帧（防止显存爆炸，与Flask一致）
IMAGE_SIZE = 224          # 帧尺寸（适配模型输入）
BATCH_SIZE = 16           # 批次大小

# API 配置 (假设 Flask Server 运行在 127.0.0.1:8000)
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}/v1/chat/completions"
# 使用基础 BERT 模型，其嵌入维度为 768
EMBEDDING_MODEL = '/root/autodl-tmp/data/roberta'
EMBEDDING_SIZE = 768

# 模型生成参数（含temperature=0.0，确保确定性输出）
GENERATION_CONFIG = {
    "temperature": 0.0,          # 你要求的确定性输出
    "max_new_tokens": 1024,      # 足够长的输出容纳JSON
    "do_sample": False,          # temperature=0.0必须关闭采样
    "pad_token_id": None,        # 后续由processor自动配置
    "eos_token_id": None,         # 后续由processor自动配置
    "max_frames": 16
}

class_names = [
'Not-Engagement',
'Barely-Engagement',
'Engagement',
'High-Engagement'
]

ENGAGEMENT_PROMPT = """You need to identify the user's engagement level in this video. The engagement levels range from low to high as four discrete values: [not engagement, barely engagement, engagement, high engagement], representing the user's engagement from completely disengaged to highly engaged. Your answer must be one of these four values, and you should not produce any other irrelevant output.

Please return your response without any Explanation in JSON format:
{{
	"Engagement_Level": "not engagement/barely engagement/engagement/high engagement"
}}
"""

# -------------------------- API 调用函数 (新增) --------------------------
def call_server_api(video_path, prompt):
    """
    调用外部 Flask 服务器的 /v1/chat/completions 接口进行推理。
    
    注意：API Server (即您提供的 Flask 腳本) 必须在运行中，
          且需要能够访问 video_path 指向的本地文件。
    """
    # 构造请求 payload (遵循 OpenAI 格式，但使用 video_path 字段)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video_path": video_path, 
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        # 传递生成参数
        "temperature": GENERATION_CONFIG["temperature"],
        "max_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample" : GENERATION_CONFIG['do_sample'],
        "max_frames": GENERATION_CONFIG['max_frames']
    }
    
    print(f"Calling API: {API_URL} for video: {os.path.basename(video_path)}")
    
    try:
        # 设置较长的超时时间，以应对大型模型推理
        response = requests.post(API_URL, json=payload, timeout=300) 
        response.raise_for_status() # 检查 HTTP 错误
        
        api_response = response.json()
        
        # 解析 API 响应的 content 字段 (标准 OpenAI 格式)
        output_text = api_response['choices'][0]['message']['content'].strip()
        
        # 解析 JSON 格式
        try:
            # 清理可能的 markdown 围栏，确保 json.loads 成功
            output_text_clean = output_text.strip().strip('`').strip('json').strip()
            engagement_result = json.loads(output_text_clean)
            return engagement_result, output_text
        
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from API response for video {os.path.basename(video_path)}. Error: {str(e)}")
            return {
                "classification": "JSON_PARSE_ERROR",
                "analysis": f"Failed to generate valid JSON. Raw model output: {output_text}"
            }, output_text

    except requests.exceptions.RequestException as e:
        print(f"Error calling API for video {os.path.basename(video_path)}: {str(e)}")
        return {
            "classification": "API_CALL_ERROR",
            "analysis": f"API call failed: {str(e)}"
        }, ""

def call_server_api_text(prompt):
    """
    调用外部 Flask 服务器的 /v1/chat/completions 接口进行文本推理（无视频）。
    
    注意：API Server (即您提供的 Flask 腳本) 必须在运行中。
    """
    # 构造请求 payload（仅保留文本类型的 content）
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}  # 只传递文本提示词
                ]
            }
        ],
        # 保留原生成参数（按需调整）
        "temperature": GENERATION_CONFIG["temperature"],
        "max_tokens": GENERATION_CONFIG["max_new_tokens"],
        "do_sample": GENERATION_CONFIG['do_sample'],
    }
    
    # 日志简化（只打印 prompt 相关，截取前50字符避免过长）
    prompt_log = prompt[:50] + "..." if len(prompt) > 50 else prompt
    print(f"Calling API: {API_URL} with prompt: {prompt_log}")
    
    try:
        # 保持原超时和请求逻辑
        response = requests.post(API_URL, json=payload, timeout=300)
        response.raise_for_status()  # 检查 HTTP 错误
        
        api_response = response.json()
        
        # 解析响应（逻辑不变）
        output_text = api_response['choices'][0]['message']['content'].strip()
        
        try:
            # 清理 markdown 围栏，解析 JSON
            output_text_clean = output_text.strip().strip('`').strip('json').strip()
            engagement_result = json.loads(output_text_clean)
            return engagement_result, output_text
        
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from API response. Prompt: {prompt_log}. Error: {str(e)}")
            return {
                "classification": "JSON_PARSE_ERROR",
                "analysis": f"Failed to generate valid JSON. Raw model output: {output_text}"
            }, output_text

    except requests.exceptions.RequestException as e:
        print(f"Error calling API. Prompt: {prompt_log}. Error: {str(e)}")
        return {
            "classification": "API_CALL_ERROR",
            "analysis": f"API call failed: {str(e)}"
        }

# -------------------------- 主函数（批量处理视频，核心逻辑对齐Flask） --------------------------
def main():

    class Args:
        dataset = "DAiSEE"
    
    args = Args()

    cudnn.benchmark = True

    train_dataset = train_data_loader(
        list_file=TRAIN_LIST_FILE,  # 测试集文件列表
        sl_file=SL_FILE,
        num_segments=16,
        duration=1,
        image_size=112,
        args=args
    )
    test_dataset = test_data_loader(
        list_file=TEST_LIST_FILE,  # 测试集文件列表
        sl_file=SL_FILE,
        num_segments=16,
        duration=1,
        image_size=112,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 测试阶段禁用shuffle，便于定位问题
        num_workers=8,   # 与你Flask代码一致，避免多线程视频读取冲突
        pin_memory=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 测试阶段禁用shuffle，便于定位问题
        num_workers=8,   # 与你Flask代码一致，避免多线程视频读取冲突
        pin_memory=True
    )

    print(f"Dataset initialized successfully: {len(test_dataset)} videos, batch size = {BATCH_SIZE}")
    print("=" * 60)

    # train for one epoch
    inference(train_loader, log_txt_path)

    return


import csv
def inference(train_loader, log_txt_path):

    losses = AverageMeter('Qwen_Loss', ':.4f')
    top1 = AverageMeter('Qwen_Accuracy', ':6.4f')
    top_score = AverageMeter('Qwen_Accuracy_score', ':6.4f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1, top_score],
                             log_txt_path=log_txt_path)

    csv_output_path = f"/root/autodl-tmp/code/qwen_engagement/qwen_test_video_daisee.csv"
    
    reverse_class_mapping_lower = {
        "not engagement": 0,
        "barely engagement": 1,
        "engagement": 2,
        "high engagement": 3
    }
    analysis_data = {}
    
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["video_path", "Engagement_Level"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        print(f"结果保存到: {csv_output_path}")

        # 使用 _ 忽略不需要的 data_loader 返回值
        for batch_idx, (video_paths, labels, extract_features, signals_data_values_sl) in enumerate(train_loader):
            for video_path, label, current_features, current_signal in zip(video_paths, labels, extract_features, signals_data_values_sl):
                result_entry = {"video_path": video_path, "Engagement_Level": ""}
                try:
                    video_files = [f for f in os.listdir(video_path) if f.lower().endswith(('.avi', '.mp4'))]
                
                    if not video_files:
                        print(f"\nWarning: No video file found in {video_path}. Skipping.")
                        continue
                    
                    if len(video_files) > 1:
                        print(f"\nWarning: Multiple videos found in {video_path}. Using the first one: {video_files[0]}")
                    
                    video_full_path = os.path.join(video_path, video_files[0])
                    video_full_path = os.path.dirname(video_full_path)
                    video_full_path_avi = os.path.join(video_full_path, video_files[0])

                   
                    final_analysis, _ = call_server_api(
                        video_path=video_full_path_avi,
                        prompt=ENGAGEMENT_PROMPT
                    )
                    print("**********************************************************")
                    
                    engagement_level = final_analysis.get("Engagement_level", "N/A")
                    
                    ###################################################################################################################################################
                    engagement_level = engagement_level.lower()
                    predicted_label = reverse_class_mapping_lower.get(engagement_level, -1)

                    is_correct = 1 if label == predicted_label else 0
                    loss_value = 1 - is_correct                     

                   
                    result_entry["Engagement_Level"] = engagement_level
                    

                except Exception as e:
                    print(f"\n處理 {video_path} 時發生嚴重錯誤: {e}")
                    traceback.print_exc()
                    result_entry["Engagement_Level"] = None

                writer.writerow(result_entry)

                csvfile.flush()

            progress.display(batch_idx)
                    
   
    print(f"\n所有结果已保存在 {csv_output_path}")
    return

def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        # with open(self.log_txt_path, 'a') as f:
        #     f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program terminated with error: {str(e)}")
        raise