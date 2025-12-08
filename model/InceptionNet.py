import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv1 = Conv2d(in_channels, 32, 3, stride=2)
        self.conv2 = Conv2d(32, 32, 3)
        self.conv3 = Conv2d(32, 64, 3, padding=1)
        
        self.branch3x3_conv = Conv2d(64, 96, 3, stride=2)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=2)
        
        self.branch7x7a = nn.Sequential(
            Conv2d(160, 64, 1),
            Conv2d(64, 64, (7,1), padding=(3,0)),
            Conv2d(64, 64, (1,7), padding=(0,3)),
            Conv2d(64, 96, 3)
        )
        
        self.branch7x7b = nn.Sequential(
            Conv2d(160, 64, 1),
            Conv2d(64, 96, 3)
        )
        
        self.branch3x3_conv2 = Conv2d(192, 192, 3, stride=2)
        self.branch3x3_pool2 = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Second block
        x1 = self.branch3x3_conv(x)
        x2 = self.branch3x3_pool(x)
        x = torch.cat((x1, x2), 1)
        
        # Third block
        x1 = self.branch7x7a(x)
        x2 = self.branch7x7b(x)
        x = torch.cat((x1, x2), 1)
        
        # Fourth block
        x1 = self.branch3x3_conv2(x)
        x2 = self.branch3x3_pool2(x)
        x = torch.cat((x1, x2), 1)
        
        return x

class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2d(384, 96, 1)
        
        self.branch3x3_1 = Conv2d(384, 64, 1)
        self.branch3x3_2 = Conv2d(64, 96, 3, padding=1)
        
        self.branch3x3dbl_1 = Conv2d(384, 64, 1)
        self.branch3x3dbl_2 = Conv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = Conv2d(96, 96, 3, padding=1)
        
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            Conv2d(384, 96, 1)
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch3x3 = Conv2d(384, 384, 3, stride=2)
        
        self.branch3x3dbl_1 = Conv2d(384, 192, 1)
        self.branch3x3dbl_2 = Conv2d(192, 224, 3, padding=1)
        self.branch3x3dbl_3 = Conv2d(224, 256, 3, stride=2)
        
        self.branch_pool = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch1x1 = Conv2d(1024, 384, 1)
        
        self.branch7x7_1 = Conv2d(1024, 192, 1)
        self.branch7x7_2 = Conv2d(192, 224, (1,7), padding=(0,3))
        self.branch7x7_3 = Conv2d(224, 256, (7,1), padding=(3,0))
        
        self.branch7x7dbl_1 = Conv2d(1024, 192, 1)
        self.branch7x7dbl_2 = Conv2d(192, 192, (7,1), padding=(3,0))
        self.branch7x7dbl_3 = Conv2d(192, 224, (1,7), padding=(0,3))
        self.branch7x7dbl_4 = Conv2d(224, 224, (7,1), padding=(3,0))
        self.branch7x7dbl_5 = Conv2d(224, 256, (1,7), padding=(0,3))
        
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            Conv2d(1024, 128, 1)
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch3x3_1 = Conv2d(1024, 192, 1)
        self.branch3x3_2 = Conv2d(192, 192, 3, stride=2)
        
        self.branch7x7x3_1 = Conv2d(1024, 256, 1)
        self.branch7x7x3_2 = Conv2d(256, 256, (1,7), padding=(0,3))
        self.branch7x7x3_3 = Conv2d(256, 320, (7,1), padding=(3,0))
        self.branch7x7x3_4 = Conv2d(320, 320, 3, stride=2)
        
        self.branch_pool = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()
        self.branch1x1 = Conv2d(1536, 256, 1)
        
        self.branch3x3_1 = Conv2d(1536, 384, 1)
        self.branch3x3_2a = Conv2d(384, 256, (1,3), padding=(0,1))
        self.branch3x3_2b = Conv2d(384, 256, (3,1), padding=(1,0))
        
        self.branch3x3dbl_1 = Conv2d(1536, 384, 1)
        self.branch3x3dbl_2 = Conv2d(384, 448, (3,1), padding=(1,0))
        self.branch3x3dbl_3 = Conv2d(448, 512, (1,3), padding=(0,1))
        self.branch3x3dbl_4a = Conv2d(512, 256, (1,3), padding=(0,1))
        self.branch3x3dbl_4b = Conv2d(512, 256, (3,1), padding=(1,0))
        
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            Conv2d(1536, 256, 1)
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_4a(branch3x3dbl),
            self.branch3x3dbl_4b(branch3x3dbl)
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionV4(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3):
        super(InceptionV4, self).__init__()
        self.stem = Stem(in_channels)
        
        self.inception_a = nn.Sequential(
            InceptionA(),
            InceptionA(),
            InceptionA(),
            InceptionA()
        )
        
        self.reduction_a = ReductionA()
        
        self.inception_b = nn.Sequential(
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB()
        )
        
        self.reduction_b = ReductionB()
        
        self.inception_c = nn.Sequential(
            InceptionC(),
            InceptionC(),
            InceptionC()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1536, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def inception_v4(num_classes=4, in_channels=3):
    return InceptionV4(num_classes=num_classes, in_channels=in_channels)


class VideoLevelInceptionNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=3):
        super(VideoLevelInceptionNet, self).__init__()
        # 基础特征提取
        self.inception = InceptionV4(num_classes=1536, in_channels=in_channels)  # 修改输出维度
        
        # 视频级别的时序聚合
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.temporal_attention = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1536, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: 视频输入 [B, T, C, H, W]
        """
        B, T = x.shape[:2]
        features = []
        
        # 1. 提取每帧特征
        for t in range(T):
            feat = self.inception.stem(x[:, t])
            feat = self.inception.inception_a(feat)
            feat = self.inception.reduction_a(feat)
            feat = self.inception.inception_b(feat)
            feat = self.inception.reduction_b(feat)
            feat = self.inception.inception_c(feat)
            feat = self.inception.avgpool(feat)
            feat = feat.view(feat.size(0), -1)  # [B, 1536]
            features.append(feat)
            
        features = torch.stack(features, dim=1)  # [B, T, 1536]
        
        # 2. 时序注意力聚合
        attention_weights = self.temporal_attention(features)  # [B, T, 1]
        video_features = (features * attention_weights).sum(1)  # [B, 1536]
        video_features = self.dropout(video_features)
        
        # 3. 分类
        output = self.fc(video_features)
        return output