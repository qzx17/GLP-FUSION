import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from .video_transform import *
import numpy as np
from natsort import natsorted
import pandas as pd
from .data_augmentation import interaug
i=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VideoRecord(object):
    def __init__(self, row):
        self._data=row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])
    
    @property
    def csv_file(self):
        return self._data[3]
    
    @property
    def sl_csv_file(self):
        return self._data[4]


class VideoDataset(data.Dataset):
    def __init__(self, list_file, sl_file, num_segments, duration, mode, transform, image_size):

        self.list_file = list_file
        self.sl_file = sl_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.video_list = []  # 初始化 video_list
        self._parse_list()
        pass

    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, class_idx]
        #
        with open(self.list_file, 'r') as file:
            lines = file.readlines()
        tmp = [line.strip().split(' ') for line in lines]
        tmp = [item for item in tmp if len(item) == 4]  # 确保每行有3个元素

        for item in tmp:
            video_path = item[0]
            num_frames = 256  # 使用固定帧数
            class_idx = item[2]
            csv_file_path = item[3]  # 添加CSV文件路径
            sl_csv_file_path = self.sl_file
            video_info = (video_path, num_frames, class_idx, csv_file_path, sl_csv_file_path)
            self.video_list.append(VideoRecord(video_info))
        print(('video number: %d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        #
        # Split all frames into seg parts, then select frame in each part randomly
        #
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames),
                             'edge')
        return offsets

    def _get_test_indices(self, record):

        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))
        video_frames_path=natsorted(video_frames_path)
        sl_csv_data = pd.read_csv(record.sl_csv_file)
        behavioral_features_sl = ["heart_rates", "p2p_intervals", "sys_peaks", "dys_peaks"]
        # 读取CSV文件
        # csv_data = pd.read_csv(record.csv_file)
        assert os.path.isfile(record.csv_file), f"CSV file not found: {record.csv_file}"
        try:
            csv_data = pd.read_csv(record.csv_file)
            # 提取文件名（不包括路径）
            subject_id = os.path.basename(record.csv_file).split('.')[0]
            #EmotiW
            # if (subject_id + '.mp4') in sl_csv_data['video_names'].values or (subject_id + '.avi') in sl_csv_data['video_names'].values:
            #    matching_rows = sl_csv_data[(sl_csv_data['video_names'] == subject_id + '.mp4') | (sl_csv_data['video_names'] == subject_id + '.avi')].head(280)
            #    signals_data_values_sl = matching_rows[behavioral_features_sl].values
            if (subject_id + '.mp4') in sl_csv_data['all_ids'].values or (subject_id + '.avi') in sl_csv_data['all_ids'].values:
                matching_rows = sl_csv_data[(sl_csv_data['all_ids'] == subject_id + '.mp4') | (sl_csv_data['all_ids'] == subject_id + '.avi')]
                signals_data_values_sl = matching_rows[behavioral_features_sl].values
            else:
                print(f"No matching video file found for subject {subject_id}")
                signals_data_values_sl = [0,0,0,0]
                signals_data_values_sl = np.array(signals_data_values_sl)
                signals_data_values_sl=signals_data_values_sl.reshape(1,-1)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {record.csv_file}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {record.csv_file}. Details: {e}")
        # print(len(video_frames_path))
        # print(len(indices))
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                 # 打开图像文件
                img_path = os.path.join(video_frames_path[p])
                img = Image.open(img_path).convert('RGB')
                # # 调整图像尺寸到224x224
                # img = img.resize((224, 224), Image.BICUBIC)
                # 将调整尺寸后的图像添加到列表
                seg_imgs = [img]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        # print(len(images))
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        # print(images.shape)
        # csv_file=record.csv_file
        # df=pd.read_csv(csv_file)
        # 提取所需的列
        #EmotiW
        # extract_features=csv_data[[' gaze_0_x',' gaze_0_y',' gaze_1_x',' gaze_1_y']].values
        # extract_features=csv_data[[' pose_Rx',' pose_Ry',' gaze_0_x',' gaze_0_y',' gaze_1_x',' gaze_1_y']].values
        #DAiSEE
        # extract_features=csv_data[['gaze_0_x','gaze_0_y','gaze_1_x','gaze_1_y']].values
        extract_features=csv_data[['pose_Rx','pose_Ry','gaze_0_x','gaze_0_y','gaze_1_x','gaze_1_y']].values
        signals_data_values_sl=signals_data_values_sl.astype(np.float32)
        signals_data_values_sl=np.expand_dims(signals_data_values_sl,axis=1)
        # 将提取的 NumPy 数组转换为 float32 类型
        extract_features = extract_features.astype(np.float32)
        
        extract_features = np.expand_dims(extract_features, axis=1)
        return images, record.label, extract_features, signals_data_values_sl

    def __len__(self):
        return len(self.video_list)


def train_data_loader(list_file, sl_file, num_segments, duration, image_size, args):
    if args.dataset == "DAiSEE":
        train_transforms = torchvision.transforms.Compose([
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            # GroupNormalize(mean=[0.334, 0.258, 0.242], std=[0.250, 0.194, 0.192]),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            ])
    if args.dataset == "EmotiW":
        train_transforms = torchvision.transforms.Compose([
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            # GroupNormalize(mean=[0.338, 0.248, 0.235], std=[0.254, 0.202, 0.195]),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    train_data = VideoDataset(list_file=list_file,
                              sl_file=sl_file,
                              num_segments=num_segments,
                              duration=duration,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,)
    return train_data


def test_data_loader(list_file, sl_file, num_segments, duration, image_size):
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor(),
                                                     GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    #  GroupNormalize(mean=[0.334, 0.258, 0.242], std=[0.250, 0.194, 0.192]),
                                                    #  GroupNormalize(mean=[0.338, 0.248, 0.235], std=[0.254, 0.202, 0.195]),
                                                     ])

    test_data = VideoDataset(list_file=list_file,
                             sl_file=sl_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data
