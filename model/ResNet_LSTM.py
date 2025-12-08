import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50LSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=3, dropout=0.25, bidirectional=True):
        """
        Args:
            num_classes: 分类数量
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: dropout率
            bidirectional: 是否使用双向LSTM
        """
        super(ResNet50LSTM, self).__init__()

        # ResNet18部分
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # 移除全连接层

        # LSTM部分
        self.lstm = nn.LSTM(
            # input_size=512,  # ResNet18的输出特征维度
            input_size=2048,  # ResNet50的输出特征维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, 
            num_classes
        )

    def forward(self, x):
        # x: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.size()
        
        x = x.view(-1, channels, height, width)  # (batch_size * num_frames, channels, height, width)
        x = self.resnet(x)  # (batch_size * num_frames, 512)
        
        # 重塑张量以准备输入LSTM
        x = x.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 512)
        
        # 通过LSTM处理序列
        x, _ = self.lstm(x)  # (batch_size, num_frames, hidden_size*2)
        
        # 使用最后一个时间步的输出
        x = x[:, -1, :]  # (batch_size, hidden_size*2)
        
        # Dropout
        x = self.dropout(x)
        
        # 输出层
        x = self.output(x)  # (batch_size, num_classes)
        
        return x

    def _init_weights(self):
        """
        初始化LSTM和线性层的权重
        """
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)