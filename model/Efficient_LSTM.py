import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
class EfficentLSTM(nn.Module):
    def __init__(self, model_name="efficientnet-b7", hidden_size=1024, dropout=0.2, num_classes=4):
        super().__init__()
        # 联网下载预训练模型
        self.base_model = EfficientNet.from_pretrained(model_name)
        
        # 动态获取特征维度
        self.input_dim = self.base_model._fc.in_features
        print(f"[INFO] EfficientNet特征维度: {self.input_dim}")
        
        # 删除分类头，只保留特征提取
        self.base_model._fc = nn.Identity()

        self.lstm = nn.LSTM(input_size=self.input_dim, batch_first=True, hidden_size=hidden_size)

        self.dropout = nn.Dropout(dropout)


        self.out_fc = nn.Sequential(nn.Linear(hidden_size, num_classes))
    def forward(self, x):
        n, t, c, h, w = x.shape
        x = x.contiguous().view(-1, c, h, w)
        x = self.base_model(x)
        x = x.contiguous().view(n, -1, self.input_dim)
        output, _ = self.lstm(x)
        x = output[:, -1, :]  # (n, 1024)
        x = self.dropout(x)
        x = self.out_fc(x)
        return x