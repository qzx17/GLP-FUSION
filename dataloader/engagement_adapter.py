"""
数据适配器：将img_dataloader的数据格式适配到MGAFR融合模型
保持与MGAFR-main相同的模态：视觉 + 文本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EngagementDataAdapter:
    """
    适配器：将img_dataloader输出转换为MGAFR融合模型输入
    
    img_dataloader输出：
        - images: [batch, num_frames, 3, H, W]
        - label: [batch]
        - extract_features: [batch, seq_len, 1, behavior_dim]  # 不使用
        - signals_data_values_sl: [batch, seq_len, 1, physio_dim]  # 不使用
        - analysis_embeddings: [batch, 1, 1, 768]  # 文本嵌入
    
    MGAFR模型输入：
        - frames: [batch, num_frames, 3, H, W]
        - text_embeddings: [batch, 768]
    """
    
    @staticmethod
    def adapt(dataloader_output):
        """
        适配数据格式
        
        Args:
            dataloader_output: tuple (images, label, extract_features, signals_data_values_sl, analysis_embeddings)
        
        Returns:
            adapted_data: {
                'frames': [batch, num_frames, 3, H, W],
                'text_embeddings': [batch, 768],
                'labels': [batch]
            }
        """
        images, labels, extract_features, signals_data_values_sl, analysis_embeddings = dataloader_output
        
        # 1. 视觉特征：保持原样
        frames = images  # [batch, num_frames, 3, H, W]
        
        # 2. 文本特征：从 [batch, 1, 1, 768] -> [batch, 768]
        text_embeddings = analysis_embeddings.squeeze(1).squeeze(1)  # [batch, 768]
        
        # 3. 标签：确保是长整型
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        elif labels.dtype != torch.long:
            labels = labels.long()
        
        return {
            'frames': frames,
            'text_embeddings': text_embeddings,
            'labels': labels
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        自定义collate函数，用于DataLoader
        
        Args:
            batch: list of tuples from dataloader
        
        Returns:
            batched_data: dict
        """
        images_list = []
        labels_list = []
        text_embeds_list = []
        
        for sample in batch:
            images, label, _, _, analysis_embeddings = sample
            
            images_list.append(images)
            labels_list.append(label)
            text_embeds_list.append(analysis_embeddings.squeeze(1).squeeze(1))
        
        # 堆叠成batch
        frames = torch.stack(images_list, dim=0)
        labels = torch.tensor(labels_list, dtype=torch.long)
        text_embeddings = torch.stack(text_embeds_list, dim=0)
        
        return {
            'frames': frames,
            'text_embeddings': text_embeddings,
            'labels': labels
        }


class CLIPTextEmbeddingCache:
    """
    CLIP文本嵌入缓存工具
    为每个参与度级别生成并缓存文本嵌入
    """
    def __init__(self, engagement_labels, clip_model_name='/bert-based-uncase'):
        """
        Args:
            engagement_labels: list of str, 参与度级别描述
                例如：['highly engaged', 'engaged', 'barely engaged', 'not engaged']
            clip_model_name: CLIP文本编码器模型名
        """
        self.engagement_labels = engagement_labels
        self.num_classes = len(engagement_labels)
        self.embeddings = None
        self.clip_model_name = clip_model_name
    
    def generate_embeddings(self, device='cuda'):
        """
        使用CLIP生成类别文本嵌入
        
        Returns:
            embeddings: [num_classes, 768] tensor
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            
            tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name)
            model = AutoModel.from_pretrained(self.clip_model_name).to(device)
            model.eval()
            
            embeddings_list = []
            
            with torch.no_grad():
                for label_text in self.engagement_labels:
                    # 构建提示
                    prompt = f"A student who is {label_text} in the classroom"
                    
                    # 编码
                    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
                    outputs = model(**inputs)
                    
                    # 使用[CLS] token的嵌入
                    embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
                    embeddings_list.append(embedding)
            
            self.embeddings = torch.cat(embeddings_list, dim=0)  # [num_classes, 768]
            
            print(f"✅ 生成 {self.num_classes} 个参与度级别的CLIP文本嵌入")
            return self.embeddings
            
        except Exception as e:
            print(f"❌ CLIP文本嵌入生成失败：{str(e)}")
            # 返回随机嵌入作为fallback
            self.embeddings = torch.randn(self.num_classes, 768).to(device)
            return self.embeddings
    
    def get_embeddings(self, device='cuda'):
        """获取文本嵌入（如果未生成则先生成）"""
        if self.embeddings is None:
            self.generate_embeddings(device)
        return self.embeddings.to(device)


def create_engagement_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4):
    """
    创建适配MGAFR模型的DataLoader
    
    Args:
        dataset: VideoDataset实例（来自img_dataloader.py）
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
    
    Returns:
        dataloader: 适配后的DataLoader
    """
    from torch.utils.data import DataLoader
    
    adapter = EngagementDataAdapter()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=adapter.collate_fn,
        pin_memory=True
    )
    
    return dataloader
