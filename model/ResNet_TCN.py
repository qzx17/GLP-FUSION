import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torchvision.models as models

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class ResNet50TCN(nn.Module):
    def __init__(self, num_classes):
        
        super(ResNet50TCN, self).__init__()

        # ResNet50部分，使用预训练的ResNet50模型
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 移除全连接层，因为我们只关心特征提取

        # TCN部分
        num_channels = 2048  # ResNet50的特征通道数
        self.tcn = TemporalConvNet(num_inputs=num_channels, num_channels=[128, 128, 128], kernel_size=3, dropout=0.2)

        # 输出层
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch_size, num_frames, channels, height, width)

        # ResNet50部分
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # 展平时间维度
        x = self.resnet(x)
        x = x.view(batch_size, num_frames, -1)  # 重塑为: (batch_size, num_frames, 2048)
        x = x.permute(0, 2, 1)  # 调整维度以匹配TCN的输入格式: 

        # TCN部分
        x = self.tcn(x)
        # print(x.shape)
        # 在时间维度上对TCN的输出进行平均
        x = x.mean(dim=2)  # 输出: (batch_size, 128)
        # print(x.shape)
        # 输出层
        x = self.output(x)

        return x
    

class ResNet18TCN(nn.Module):
    def __init__(self, num_classes):
        
        super(ResNet18TCN, self).__init__()

        self.spatial_feat_dim = 32
        self.num_classes = 4
        self.nhid = 128
        self.levels = 8
        self.kernel_size = 7
        self.dropout = .1
        self.channel_sizes = [self.nhid] * self.levels

        self.tcn = TemporalConvNet(
            self.spatial_feat_dim,
            self.channel_sizes,
            kernel_size=self.kernel_size,
            dropout=self.dropout)
        self.linear = nn.Linear(self.channel_sizes[-1], self.num_classes)

        self.model_conv = models.resnet50(pretrained=True)
        # for param in self.model_conv.parameters():
        #     param.requires_grad = False

        num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(num_ftrs, self.spatial_feat_dim)


    def forward(self, x):
        # x: (batch_size, num_frames, channels, height, width)

        # ResNet50部分
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # 展平时间维度
        x = self.resnet(x)
        x = x.view(batch_size, num_frames, -1)  # 重塑为: (batch_size, num_frames, 2048)
        x = x.permute(0, 2, 1)  # 调整维度以匹配TCN的输入格式: 

        # TCN部分
        x = self.tcn(x)
        # print(x.shape)
        # 在时间维度上对TCN的输出进行平均
        x = x.mean(dim=2)  # 输出: (batch_size, 128)
        # print(x.shape)
        # 输出层
        x = self.output(x)

        return x