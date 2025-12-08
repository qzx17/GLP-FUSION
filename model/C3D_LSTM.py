import torch
import torch.nn as nn

class StudentEngagementNet(nn.Module):
    """
    结合C3D和LSTM的学生参与度分类网络
    输出4个类别：低参与度->高参与度
    """
    def __init__(self):
        super(StudentEngagementNet, self).__init__()
        
        # C3D部分
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        # 修改全连接层
        self.fc6 = nn.Linear(8192, 4096)
        self.relu = nn.ReLU()
        
        # LSTM部分
        self.rnn = nn.LSTM(4096, 256, 1, batch_first=True)
        
        # 分类器
        self.fc_classifier = nn.Linear(256, 4)
        self.softmax = nn.Softmax(dim=1)
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 调整输入数据形状
        x = x.permute(0, 2, 1, 3, 4)  # 从[B,T,C,H,W]变为[B,C,T,H,W]
        # C3D特征提取
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        # 展平并通过全连接层
        h = h.view(-1, 8192)
        h = self.dropout(self.relu(self.fc6(h)))
        
        # 重塑张量以适应LSTM输入
        batch_size = x.size(0)
        h = h.view(batch_size, -1, 4096)  # (batch_size, num_frames, 4096)
        
        # LSTM处理
        lstm_output, _ = self.rnn(h)
        
        # 获取最后一个时间步的输出并分类
        final_output = self.dropout(lstm_output[:, -1, :])
        logits = self.fc_classifier(final_output)
        probs = self.softmax(logits)
        
        return probs

    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

def create_model():
    """
    创建并初始化模型
    """
    model = StudentEngagementNet()
    model._initialize_weights()
    return model

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = create_model()
    
    # 生成示例输入数据
    # batch_size=32, frames=16, channels=3, height=112, width=112
    sample_input = torch.randn(16, 16, 3, 112, 112)
    # 前向传播
    output = model(sample_input)
    print(f"输入形状: {sample_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例:\n{output[0]}")  # 打印第一个样本的预测概率
