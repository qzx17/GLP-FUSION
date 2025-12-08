import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score


# -------------------------- 一、配置参数（适配无音频+输入维度） --------------------------
class Config:
    # 输入维度适配
    BATCH_SIZE = 16  # 与输入text/frames的第1维一致
    TEXT_SEQ_LEN = 16  # 文本序列长度（输入第2维）
    TEXT_HIDDEN_DIM = 768  # 文本特征维度（输入第3维）
    FRAME_NUM_PER_SAMPLE = 16  # 每个样本的帧数（frames第2维）
    FRAME_SIZE = (224, 224)  # 帧分辨率（frames第4-5维）
    # 模态参数（参考原文档，剔除音频）
    NUM_EMOTIONS = 4 
    MODALITIES = ["text", "face", "motion"]  # 无音频
    TEXT_LR = 1e-5  # 原文档文本DeBERTa学习率
    FACE_LR = 1e-3  # 原文档ResNet-50学习率
    MOTION_LR = 1e-4  # 原文档3D-CNN学习率
    EPOCHS = 10  # 统一训练轮次（可按需调整）
    MOD_ACCURACIES = torch.tensor([0.8443, 0.7048, 0.6872], dtype=torch.float32)  # 论文中 文本/面部/运动
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


# -------------------------- 二、数据集类（适配你的输入格式） --------------------------
class NoAudioMISTDataset(Dataset):
    """接收已预处理的text特征和frames特征，无需再提取原始数据（适配你的输入）"""
    def __init__(self, text_features, frame_features, labels):
        """
        Args:
            text_features:  tensor (N, 16, 768)，N=总样本数，与batch_size匹配
            frame_features: tensor (N, 16, 3, 224, 224)，N=总样本数
            labels:         tensor (N,)，情感标签（0-6）
        """
        assert len(text_features) == len(frame_features) == len(labels), "样本数不匹配"
        self.text = text_features
        self.frames = frame_features
        self.labels = labels

    def __getitem__(self, idx):
        # 1. 文本特征：直接取预处理后的特征（无需再处理）
        text_feat = self.text[idx]  # (16, 768)
        
        # 2. 面部特征：取16帧的平均（原文档4.6：ResNet-50用视频平均帧）
        frames = self.frames[idx]  # (16, 3, H, W)
        face_feat = torch.mean(frames, dim=0)  # (3, H, W)，平均后单帧
        
        # 3. 运动特征：直接使用所有16帧（不再采样）
        motion_feat = frames  # (16, 3, H, W)，直接使用16帧

        return {
            "text": text_feat,
            "face": face_feat,
            "motion": motion_feat,
            "label": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)


# -------------------------- 三、单模态模型（适配输入+无音频） --------------------------
# 1. 文本模态：基于输入特征的分类（原文档DeBERTa逻辑，输入已为特征，无需预训练模型）
class TextModel(nn.Module):
    def __init__(self, num_classes=config.NUM_EMOTIONS):
        super().__init__()
        # 原文档4.4：DeBERTa输出特征后接分类头，此处直接用输入的(16,768)特征
        self.fc_layers = nn.Sequential(
            # 先对序列维度平均池化：(16,768)→(768)（整合16个token的特征）
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # (768)
            # 原文档“随机初始化全连接层”逻辑
            nn.Linear(config.TEXT_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # 补充：防止过拟合（原文档未明确，通用实践）
            nn.Linear(256, num_classes)
        )

    def forward(self, text_feat):
        """text_feat: (batch_size, 5, 768)"""
        # 调整维度：(16,5,768)→(16,768,5)（适配AdaptiveAvgPool1d的输入格式）
        x = text_feat.permute(0, 2, 1)  # (batch_size, 768, 16)
        x = self.fc_layers(x)  # (batch_size, num_classes)
        return x


# 2. 面部模态：ResNet-50（原文档4.6，完全一致）
class FaceModel(nn.Module):
    def __init__(self, num_classes=config.NUM_EMOTIONS):
        super().__init__()
        # 加载预训练ResNet-50，微调分类头（原文档逻辑）
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)  # 替换分类头

    def forward(self, face_feat):
        """face_feat: (batch_size, 3, 224, 224)（平均帧特征）"""
        return self.resnet50(face_feat.to(config.DEVICE))  # (batch_size, num_classes)


# 3. 运动模态：3D-CNN（原文档4.7，适配16帧输入）
class MotionModel(nn.Module):
    def __init__(self, num_classes=config.NUM_EMOTIONS):
        super().__init__()
        # 输入：(batch_size, 1, 16, H, W)（1=灰度通道，16=帧数）
        self.tdcnn_layers = nn.Sequential(
            # 卷积层1：捕捉时空特征
            nn.Conv3d(
                in_channels=1, 
                out_channels=32, 
                kernel_size=(3, 5, 5),  # (时间维度, 空间维度, 空间维度)
                stride=1, 
                padding=(1, 2, 2)  # 时间维度padding=1保持16帧，空间维度补边
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # 下采样：16->8帧
            # 卷积层2：加深特征
            nn.Conv3d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1)  # 保持尺寸
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # 下采样：8->4帧
            # 卷积层3：进一步提取特征
            nn.Conv3d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=(3, 3, 3),
                stride=1,
                padding=(1, 1, 1)
            ),
            nn.ReLU(),
            # 全局平均池化
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, motion_feat):
        """motion_feat: (batch_size, 1, 16, H, W)"""
        x = self.tdcnn_layers(motion_feat.to(config.DEVICE))
        x = self.fc_layers(x)  # (batch_size, num_classes)
        return x


# -------------------------- 四、模态融合（原文档加权平均，剔除音频） --------------------------
class NoAudioMISTFusion(nn.Module):
    def __init__(self, text_model, face_model, motion_model):
        super().__init__()
        self.text_model = text_model
        self.face_model = face_model
        self.motion_model = motion_model
        # 原文档Eq.1参数：mod_j（模态准确率）、rec_ij（模态-情感召回率）
        self.mod_j = config.MOD_ACCURACIES.to(config.DEVICE)  # (3,)：文本/面部/运动
        # rec_ij：(num_emotions, num_modalities)，需从验证集计算（示例初始化为1）
        self.rec_ij = torch.ones(config.NUM_EMOTIONS, len(config.MODALITIES), device=config.DEVICE)

    def forward(self, batch):
        """
        输入batch格式：
        - batch["text"]: (batch_size, 16, 768)
        - batch["face"]: (batch_size, 3, 224, 224) - 已经平均的单帧
        - batch["motion"]: (batch_size, 16, 3, H, W) - 所有16帧，直接使用
        - batch["label"]: (batch_size,)
        """
        batch_size = batch["text"].size(0)
        
        # 处理motion特征：直接使用16帧，不再采样
        motion_frames_all = batch["motion"]  # (batch_size, 16, 3, H, W)

        # 转灰度图：对RGB三通道取平均
        motion_frames_gray = torch.mean(motion_frames_all, dim=2, keepdim=True)  # (batch_size, 16, 1, H, W)
        
        # 调整维度为3D-CNN输入格式：(batch_size, 1, 16, H, W)
        batch["motion"] = motion_frames_gray.permute(0, 2, 1, 3, 4)  # (batch_size, 1, 16, H, W)
        
        # 1. 各模态单独预测（输出logits→概率）
        text_logits = self.text_model(batch["text"])
        face_logits = self.face_model(batch["face"])
        motion_logits = self.motion_model(batch["motion"])

        # 转换为概率（p_i）
        text_probs = torch.softmax(text_logits, dim=1)  # (batch_size, num_classes)
        face_probs = torch.softmax(face_logits, dim=1)  # (batch_size, num_classes)
        motion_probs = torch.softmax(motion_logits, dim=1)  # (batch_size, num_classes)

        # 堆叠概率：(batch_size, num_emotions, num_modalities)
        probs_stack = torch.stack([text_probs, face_probs, motion_probs], dim=2)

        # 2. 计算权重：mod_j * rec_ij → (num_emotions, num_modalities)
        weights = self.mod_j * self.rec_ij  # (num_emotions, 3)
        weights = weights.unsqueeze(0)  # (1, num_emotions, 3)，匹配batch维度

        # 3. 加权平均（分子=概率×权重求和，分母=权重求和）
        numerator = torch.sum(probs_stack * weights, dim=2)  # (batch_size, num_emotions)
        denominator = torch.sum(weights, dim=2)  # (1, num_emotions)
        final_probs = numerator / denominator  # (batch_size, num_emotions)：每个样本的情感概率

        return final_probs
