import os
import io
import torch
import base64
import time
import traceback  # 1. 导入 traceback 模块（用于打印堆栈）
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoModelForImageTextToText, AutoProcessor
import numpy as np 
import cv2



# --- 配置信息 ---
# MODEL_PATH = "/root/autodl-tmp/Qwen/qwen2.5-vl-instruct"
MODEL_PATH = "/root/autodl-tmp/Qwen/qwen3-vl-8b"
HOST = "0.0.0.0"
PORT = 8000
FRAME_SAMPLING_RATE = 8 
MAX_FRAMES = 64  # 建议减少帧数，避免显存爆炸（256帧易触发OOM）


# --- 模型的全局變量 ---
model = None
processor = None


# --- Flask 應用設置 ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False 


# --- 模型加載函數 (终端打印错误详情) ---
def load_model():
    """全局加載模型和處理器。"""
    global model, processor
    try:
        print(f"\n[模型加载] 正在從 {MODEL_PATH} 加載模型...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype="auto", device_map="auto"  # 启用FP16省显存
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)  # 消除处理器警告
        print("[模型加载] 模型和處理器加載成功！")
    except Exception as e:
        # 2. 终端打印模型加载错误的完整堆栈
        print("\n" + "="*60)
        print("[模型加载 ERROR] 加载失败！详细错误：")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()  # 打印完整调用链
        print("="*60 + "\n")
        model = None
        processor = None


# --- 視頻處理函數 ---
def extract_frames_from_video(video_path, max_frames):
    """
    從視頻路徑中均勻提取指定數量的幀並返回 PIL Image 列表。

    Args:
        video_path (str): 視頻文件的完整路徑。
        max_frames (int): 希望從視頻中提取的最大幀數。

    Returns:
        list: 一個包含 PIL.Image 對象的列表。
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) 未安裝。無法處理視頻。")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到視頻文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法打開視頻文件: {video_path}")

    # 獲取視頻的總幀數
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果視頻沒有幀，直接返回空列表
    if total_frames == 0:
        cap.release()
        print(f"[视频处理] 警告: 視頻 {os.path.basename(video_path)} 不包含任何幀。")
        return []

    frames = []
    
    # --- 核心修改：動態計算要提取的幀的索引 ---
    
    # 如果視頻總幀數少於或等於目標數量，則提取所有幀
    if total_frames <= max_frames:
        indices_to_capture = range(total_frames)
    # 否則，計算出在整個視頻中均勻分佈的幀索引
    else:
        # 使用 np.linspace 生成在 [0, total_frames-1] 範圍內均勻分佈的 max_frames 個點
        indices_to_capture = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for frame_index in indices_to_capture:
        # 直接跳轉到指定的幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if ret:
            # 將 BGR 格式的幀轉換為 RGB 格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 轉換為 PIL Image 對象並添加到列表中
            img = Image.fromarray(frame_rgb)
            frames.append(img)

    cap.release()
    print(f"[视频处理] 成功從 {os.path.basename(video_path)} 提取 {len(frames)} 幀 (视频总帧数: {total_frames})")
    
    return frames


# --- 推理接口 (终端直打 500 错误详情) ---
@app.route('/v1/chat/completions', methods=['POST'])
def generate_multimodal_completion():
    if model is None or processor is None:
        return jsonify({"error": "模型未加載。請檢查伺服器終端日誌。"}), 503

    try:
        # 解析请求参数
        data = request.get_json()
        messages = data.get("messages")
        max_new_tokens = data.get("max_tokens", 1024) 
        temperature = data.get("temperature", 0.0)
        max_frames = data.get("max_frames", 16)
        do_sample = temperature > 0.001 

        if not messages:
            return jsonify({"error": "請求數據中缺少 'messages' 字段。"}), 400

        # 处理多模态消息
        processed_messages = []
        for msg_idx, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", [])
            processed_content = []
            
            if not isinstance(content, list):
                processed_content.append({"type": "text", "text": content})
            else:
                for item in content:
                    item_type = item.get("type")
                    if item_type == "image":
                        # 处理图片Base64
                        image_b64 = item.get("image") or item.get("image_url", {}).get("url")
                        if image_b64 and image_b64.startswith("data:image"):
                            image_b64 = image_b64.split(",")[1]
                        if image_b64:
                            image_data = base64.b64decode(image_b64)
                            image = Image.open(io.BytesIO(image_data))
                            processed_content.append({"type": "image", "image": image})
                        else:
                            return jsonify({"error": "指定了圖片類型，但圖片數據無效或缺失。"}), 400
                    elif item_type == "video":
                        # 处理视频路径
                        video_path = item.get("video_path")
                        if not video_path:
                            return jsonify({"error": "指定了視頻類型，但未提供 'video_path'。"}), 400
                        video_frames = extract_frames_from_video(video_path, max_frames)
                        for frame in video_frames:
                            processed_content.append({"type": "image", "image": frame})
                    elif item_type == "text":
                        processed_content.append(item)

            processed_messages.append({"role": role, "content": processed_content})

    except (RuntimeError, FileNotFoundError, IOError) as e:
        # 400错误（客户端问题），简单打印
        print(f"\n[请求错误] 客户端数据错误: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # 400类未预期错误，打印堆栈
        print("\n" + "="*60)
        print("[请求处理 ERROR] 格式无效或数据处理错误！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({"error": f"請求格式無效或數據處理錯誤: {str(e)}"}), 400

    # --- 2. 分词/输入准备（500错误点1）---
    try:
        print(f"\n[推理准备] 开始分词（温度: {temperature}, 最大token: {max_new_tokens}）")
        inputs = processor.apply_chat_template(
            processed_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        print(f"[推理准备] 分词完成，输入shape: {inputs.input_ids.shape}")

    except Exception as e:
        # 3. 终端打印分词阶段500错误详情
        print("\n" + "="*60)
        print("[推理准备 ERROR] 分词/输入准备失败！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()  # 关键：打印完整调用链
        print("="*60 + "\n")
        return jsonify({"error": f"分詞/輸入準備時出錯: {str(e)}"}), 500

    # --- 3. 模型生成（500错误点2）---
    try:
        print("[模型生成] 开始推理...")
        with torch.no_grad():  # 禁用梯度，省显存
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # 裁剪并解码结果
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        print(f"[模型生成] 推理完成，生成文本长度: {len(output_text)} 字符")

    except Exception as e:
        # 4. 终端打印生成阶段500错误详情
        print("\n" + "="*60)
        print("[模型生成 ERROR] 推理失败！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("错误堆栈:")
        traceback.print_exc()  # 打印完整调用链（哪行代码、哪个函数出错）
        print("="*60 + "\n")
        return jsonify({"error": f"模型生成時出錯: {str(e)}"}), 500
    
    # --- 4. Token计数（非关键，错误仅警告）---
    try:
        prompt_for_counting = processor.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_tokens = len(processor.tokenizer.encode(prompt_for_counting))
        if processor.tokenizer.pad_token is None:
             processor.tokenizer.pad_token = processor.tokenizer.eos_token
        completion_tokens = len(processor.tokenizer.encode(output_text, add_special_tokens=False))
        
    except Exception as e:
        print(f"\n[Token计数 WARNING] 计算失败: {str(e)}")
        prompt_tokens = 0
        completion_tokens = 0
        
    # --- 5. 构建响应 ---
    response = {
        "id": f"qwen-vl-chat-{int(time.time())}",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text 
                },
                "finish_reason": "stop"
            }
        ],
        "model": "qwen2.5-vl-instruct",
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    return jsonify(response)


if __name__ == "__main__":
    print("="*60)
    print("Qwen3-VL (Image-Only) 伺服器啟動中...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"服务地址: http://{HOST}:{PORT}/v1/chat/completions")
    print("="*60)
    load_model()
    if model and processor:
        print("\n服务器已启动，等待请求...")
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    else:
        print("\n服务器启动失败，模型未加载！")
