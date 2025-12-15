import os
import numpy as np
import torch
from PIL import Image
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import json
from tqdm import tqdm

import re

def load_images_from_folder(folder_path, num_frames=280, sample_frames=16):

    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bmp')])
    
    indices = np.linspace(0, num_frames - 1, sample_frames, dtype=int)
    selected_images = [Image.open(image_files[i]).convert("RGB") for i in indices]
    return selected_images
def load_dataset(file_path):

    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            video_folder = parts[0]
            label = int(parts[2])  # 参与度标签
            data_list.append((video_folder, label))
    return data_list
data_file = "EmotiW_Test.txt"  #
dataset = load_dataset(data_file)

device = "cuda:4"
model_path = "video-llava-7b-hf/"  # 你的本地模型路径
save_file = "emotiw_llava.txt"
model = VideoLlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
processor = VideoLlavaProcessor.from_pretrained(model_path)

# 构造 Prompt
prompt = "USER: <video> You need to identify the user's engagement level in this video. The engagement levels range from low to high as four discrete values: [0, 1, 2, 3], representing the user's engagement from completely disengaged to highly engaged. Your answer must be one of these four values, and you should not produce any other irrelevant output. ASSISTANT:"

max_attempts = 5
valid_responses = {"0", "1", "2", "3"}


for video_folder, label in tqdm(dataset):
    clip = load_images_from_folder(video_folder, sample_frames=16)


    tokenized_inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(device)
    pred = None
    for attempt in range(max_attempts):
        generate_ids = model.generate(**tokenized_inputs, max_new_tokens=5)
        output_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        match = re.search(r"\b([0123])\b\s*$", output_text)  # 匹配最后一个有效数字
        if match:
            pred = output_text


            break  # 输出符合要求，退出循环
        else:
            print(f"...")
        # generate_ids = model.generate(**tokenized_inputs, max_length=10)
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(response)
    if pred == None:
        pred = '4'
    data = {}
    data['gt'] = label
    data['pred'] = pred
    with open(save_file, 'a') as f:

        f.write(json.dumps(data) + '\n')
