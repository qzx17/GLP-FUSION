import os
import io
import torch
import base64
import time
import traceback
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- 配置信息 ---
MODEL_PATH = "/root/autodl-tmp/Qwen/qwen3-vl-8b"
HOST = "0.0.0.0"
PORT = 8000

# --- 模型的全局變量 ---
model = None
processor = None

# --- Flask 應用設置 ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# --- 模型加載函數 ---
def load_model():

    global model, processor
    try:
        print(f"\n[模型加载] 正在从 {MODEL_PATH} 加載模型...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
        print("[模型加载] 模型和处理器加載成功！")
    except Exception as e:
        print("\n" + "="*60)
        print("[模型加载 ERROR] 加载失败！详细错误：")
        traceback.print_exc()
        print("="*60 + "\n")
        model = None
        processor = None

# --- 推理接口 ---
@app.route('/v1/chat/completions', methods=['POST'])
def generate_multimodal_completion():
    if model is None or processor is None:
        return jsonify({"error": "模型未加載。请检查服务器终端日志。"}), 503

    try:
        # 1. 解析请求参数
        data = request.get_json()
        messages = data.get("messages")
        max_new_tokens = data.get("max_tokens", 1024)
        temperature = data.get("temperature", 0.0)
        do_sample = temperature > 0.001

        if not messages:
            return jsonify({"error": "請求數據中缺少 'messages' 字段。"}), 400

        # 2. 处理多模态消息 (簡化版：只處理圖像和文本)
        processed_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", [])
            processed_content = []
            
            if not isinstance(content, list):
                
                processed_content.append({"type": "text", "text": content})
            else:
                for item in content:
                    item_type = item.get("type")
                    if item_type == "image":
                        image_b64 = item.get("image") or item.get("image_url", {}).get("url")
                        if image_b64 and image_b64.startswith("data:image"):
                            image_b64 = image_b64.split(",")[1]
                        
                        if image_b64:
                            image_data = base64.b64decode(image_b64)
                            image = Image.open(io.BytesIO(image_data))
                            processed_content.append({"type": "image", "image": image})
                        else:
                            return jsonify({"error": "指定了图片，但图片数据无效或缺失。"}), 400
                    elif item_type == "text":
                        processed_content.append(item)
            
            processed_messages.append({"role": role, "content": processed_content})

    except Exception as e:
        print("\n" + "="*60)
        print("[请求处理 ERROR] 格式无效或数据处理错误！")
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({"error": f"请求格式无效或数据处理错误: {str(e)}"}), 400

    # 3. 分词/输入准备
    try:
        inputs = processor.apply_chat_template(
            processed_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
    except Exception as e:
        print("\n" + "="*60)
        print("[推理准备 ERROR] 分词/输入准备失败！")
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({"error": f"分词/准备时出错: {str(e)}"}), 500

    # 4. 模型生成
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    except Exception as e:
        print("\n" + "="*60)
        print("[模型生成 ERROR] 推理失败！")
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({"error": f"模型生成时出错: {str(e)}"}), 500
    
    # 5. Token计数
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
        
    # 6. 构建响应
    response = {
        "id": f"qwen-vl-chat-{int(time.time())}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
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