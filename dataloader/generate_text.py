import os
import cv2
import torch
import json
from PIL import Image
import torch.backends.cudnn as cudnn
from dataloader.video_dataloader import train_data_loader, test_data_loader
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import requests
import base64
import shutil
import traceback
import io
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
BATCH_SIZE = 1           # 批次大小

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

ENGAGEMENT_ANALYSIS_PROMPT = """You are a researcher in the field of educational psychology, specializing in analyzing classroom engagement by observing students' facial features and postures.
Please base on the provided target image (the image needs to clearly show the student's facial and head posture), and strictly follow the following 5 preset dimensions to accurately identify the specific performance of the student in the image and write a description of the student's classroom engagement. The specific requirements are as follows:

Describe from the following five dimensions:
1. **Gaze Direction**: Choose from "Center of gaze, Gaze upward, Gaze downward, Gaze leftward, Gaze rightward, Oblique upward, Oblique downward".
2. **Eyelid State**: Choose from "Eyes normally open, Slightly squinted, Eyelids drooping".
3. **Head Posture**: Choose from "Head upright, Leaning forward, Leaning to one side, Leaning backward, Propping head with hand".
4. **Mouth Movement**: Choose from "Corners of the mouth turned up, Corners of the mouth turned down, Corners of the mouth flat".
5. **Emotional Change**: Choose from "Happy, Confused, Sad, Impatient, Serious, Flat".

Please output content in JSON format:
{{
	"Gaze Direction": "",
	"Eyelid State": "",
	"Head Posture": "",
	"Mouth Movement": "",
	"Emotional Change": ""
}}
"""


def extract_frames_from_video(video_path, max_frames=5):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    frames = []
    if total_frames <= max_frames:
        indices_to_capture = range(total_frames)
    else:
        indices_to_capture = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for frame_index in indices_to_capture:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frames.append(img)

    cap.release()
    print(f"[视频处理] 从 {os.path.basename(video_path)} 提取 {len(frames)} 帧 (总帧数: {total_frames})")
    return frames



# -------------------------- API 调用函数 (新增) --------------------------
def call_server_api(image_pil, prompt):

    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    payload = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": img_b64}]}],
        "max_tokens": 1024, "temperature": 0.0
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        clean_content = content.strip().lstrip('```json').rstrip('```').strip()
        return json.loads(clean_content), content
    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        return None, str(e)


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
        shuffle=False,  # 禁用shuffle，便于定位问题
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
                             [losses, top1, top_score], # 只顯示 Qwen 的指標
                             log_txt_path=log_txt_path)


    csv_output_path = f"/root/autodl-tmp/code/qwen_engagement/qwen_train_frames.csv"

    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:

        fieldnames = ["video_path", "per_frame_analysis"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        print(f"結果將被實時保存到: {csv_output_path}")

        for batch_idx, (video_paths, labels, extract_features, signals_data_values_sl) in enumerate(train_loader):
            for video_path, ground_truth_label_tensor, current_features, current_signal in zip(video_paths, labels, extract_features, signals_data_values_sl):
                result_entry = {"video_path": video_path, "per_frame_analysis": ""}
                try:

                    video_files = [f for f in os.listdir(video_path) if f.lower().endswith(('.avi', '.mp4'))]
                
                    if not video_files:
                        print(f"\nWarning: No video file found in {video_path}. Skipping.")
                        continue
                    
                    if len(video_files) > 1:
                        print(f"\nWarning: Multiple videos found in {video_path}. Using the first one: {video_files[0]}")
                    
                    video_full_path = os.path.join(video_path, video_files[0])
    
                    frames = extract_frames_from_video(video_full_path, max_frames=16)
                    if not frames:
                        error_msg = json.dumps([{"error": "FRAME_EXTRACTION_FAILED"}])
                        result_entry["per_frame_analysis"] = error_msg
                        writer.writerow(result_entry)
                        csvfile.flush() # 強制寫入磁盤
                        continue

                    per_frame_results = []
                    for frame in frames:
                        parsed_json, raw_text = call_server_api(frame, ENGAGEMENT_ANALYSIS_PROMPT)
                        if parsed_json:
                            per_frame_results.append(parsed_json)
                        else:
                            per_frame_results.append({"error": "API_CALL_OR_PARSE_FAILED", "raw_output": raw_text})

                    result_entry["per_frame_analysis"] = json.dumps(per_frame_results, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"\n處理 {video_path} 時發生嚴重錯誤: {e}")
                    traceback.print_exc()
                    result_entry["per_frame_analysis"] = json.dumps([{"error": "FATAL_ERROR", "details": str(e)}])

                writer.writerow(result_entry)
                csvfile.flush()

    print(f"\n所有視頻處理完畢。最終結果已保存在 {csv_output_path}")
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
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'




if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program terminated with error: {str(e)}")
        raise