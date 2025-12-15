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
import csv
from transformers import AutoTokenizer, AutoModel
import pickle
import hashlib
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
    def __init__(self, list_file, sl_file, text_embedding_dict, num_segments, duration, mode, transform, image_size, pre_extracted_features_path=None):

        self.list_file = list_file
        self.sl_file = sl_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.video_list = []  # 初始化 video_list
        self.text_embedding_dict = self._load_embedding_dict(text_embedding_dict)
        
        # 新增：预提取特征路径
        self.pre_extracted_features = None
        if pre_extracted_features_path and os.path.exists(pre_extracted_features_path):
            print(f"加载预提取视觉特征: {pre_extracted_features_path}")
            with open(pre_extracted_features_path, 'rb') as f:
                self.pre_extracted_features = pickle.load(f)
            print(f"预提取特征加载成功，共 {len(self.pre_extracted_features)} 个样本")
        elif pre_extracted_features_path:
            print(f"预提取特征文件不存在: {pre_extracted_features_path}")
        
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

    def _load_embedding_dict(self, dict_path):
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"文本嵌入字典不存在！请先运行生成，路径：{dict_path}")
        try:
            with open(dict_path, 'rb') as f:
                embedding_dict = pickle.load(f)
            print(f"从 {dict_path} 加载文本嵌入字典（共 {len(embedding_dict)} 个嵌入）")
            return embedding_dict
        except Exception as e:
            raise ValueError(f"❌ 加载嵌入字典失败：{str(e)}")

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
        #
        # Split all frames into seg parts, then select frame in the mid of each part
        #
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames),
                             'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)


    def get(self, record, indices):
        # 新增：如果使用预提取特征
        if self.pre_extracted_features is not None:
            # 直接从预提取特征中加载
            if record.path in self.pre_extracted_features:
                # 预提取特征是 (num_frames, 2048) 的 numpy 数组
                visual_features = self.pre_extracted_features[record.path]
                
                # 转换为 torch.Tensor
                visual_features = torch.from_numpy(visual_features).float()
                
                # 如果需要采样特定帧
                if len(indices) < visual_features.shape[0]:
                    # 采样指定帧
                    sampled_features = visual_features[indices]
                else:
                    # 如果索引超出范围，重复最后一帧
                    sampled_features = []
                    for idx in indices:
                        if idx < visual_features.shape[0]:
                            sampled_features.append(visual_features[idx])
                        else:
                            sampled_features.append(visual_features[-1])
                    sampled_features = torch.stack(sampled_features)
                
                # 添加 batch 维度: (num_segments, 2048) -> (1, num_segments, 2048)
                visual_features_batch = sampled_features.unsqueeze(0)
            else:
                print(f"未找到预提取特征: {record.path}")
                # 回退到原始图像处理
                return self._get_from_images(record, indices)
            
            # 保持原有的CSV特征处理逻辑
            sl_csv_data = pd.read_csv(record.sl_csv_file)
            behavioral_features_sl = ["heart_rates", "p2p_intervals", "sys_peaks", "dys_peaks"]
            
            subject_id = os.path.basename(record.csv_file).split('.')[0]
            try:
                csv_data = pd.read_csv(record.csv_file)
                #EmotiW
                # if (subject_id + '.mp4') in sl_csv_data['video_names'].values or (subject_id + '.avi') in sl_csv_data['video_names'].values:
                #    matching_rows = sl_csv_data[(sl_csv_data['video_names'] == subject_id + '.mp4') | (sl_csv_data['video_names'] == subject_id + '.avi')].head(280)
                #    signals_data_values_sl = matching_rows[behavioral_features_sl].values
                if (subject_id + '.mp4') in sl_csv_data['all_ids'].values or (subject_id + '.avi') in sl_csv_data['all_ids'].values:
                    matching_rows = sl_csv_data[(sl_csv_data['all_ids'] == subject_id + '.mp4') | (sl_csv_data['all_ids'] == subject_id + '.avi')]
                    signals_data_values_sl = matching_rows[behavioral_features_sl].values
                else:
                    print(f"No matching video file found for subject {subject_id}")
                    signals_data_values_sl = np.array([0,0,0,0]).reshape(1,-1)

                # 用subject_id查pkl字典
                if subject_id in self.text_embedding_dict:
                    analysis_embeddings = self.text_embedding_dict[subject_id]
                else:
                    print(f"No embedding found for subject {subject_id}")
                    analysis_embeddings = np.zeros(768, dtype=np.float32)  # 空嵌入用全0
                # 保持形状：(1, 768)
                analysis_embeddings = np.expand_dims(analysis_embeddings, axis=0)
            except Exception as e:
                raise ValueError(f"CSV处理失败：{str(e)}")
            
            # 3. 特征格式处理（完全不变）
            #EmotiW
            # extract_features=csv_data[[' pose_Rx',' pose_Ry',' gaze_0_x',' gaze_0_y',' gaze_1_x',' gaze_1_y']].values.astype(np.float32)
            extract_features = csv_data[['pose_Rx','pose_Ry','gaze_0_x','gaze_0_y','gaze_1_x','gaze_1_y']].values.astype(np.float32)
            signals_data_values_sl = signals_data_values_sl.astype(np.float32)
            signals_data_values_sl = np.expand_dims(signals_data_values_sl, axis=1)
            extract_features = extract_features.astype(np.float32)        
            extract_features = np.expand_dims(extract_features, axis=1)

            analysis_embeddings = analysis_embeddings.astype(np.float32)
            analysis_embeddings = np.expand_dims(analysis_embeddings, axis=1)
            
            # 返回格式: (batch=1, num_segments, 2048), label, 其他特征...
            return visual_features_batch, record.label, extract_features, signals_data_values_sl, analysis_embeddings
        else:
            # 回退到原始图像处理
            return self._get_from_images(record, indices)
    
    def _get_from_images(self, record, indices):
        # 1. 直接获取该样本的文件夹地址
        video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))
        video_frames_path=natsorted(video_frames_path)

        # 2. 保留原有的CSV特征处理逻辑
        sl_csv_data = pd.read_csv(record.sl_csv_file)
        behavioral_features_sl = ["heart_rates", "p2p_intervals", "sys_peaks", "dys_peaks"]

        subject_id = os.path.basename(record.csv_file).split('.')[0]
        try:
            csv_data = pd.read_csv(record.csv_file)
            #EmotiW
            if (subject_id + '.mp4') in sl_csv_data['video_names'].values or (subject_id + '.avi') in sl_csv_data['video_names'].values:
               matching_rows = sl_csv_data[(sl_csv_data['video_names'] == subject_id + '.mp4') | (sl_csv_data['video_names'] == subject_id + '.avi')].head(280)
               signals_data_values_sl = matching_rows[behavioral_features_sl].values
            else:
                print(f"No matching video file found for subject {subject_id}")
                signals_data_values_sl = np.array([0,0,0,0]).reshape(1,-1)

            # 用subject_id查pkl字典
            if subject_id in self.text_embedding_dict:
                analysis_embeddings = self.text_embedding_dict[subject_id]
            else:
                print(f"No embedding found for subject {subject_id}")
                analysis_embeddings = np.zeros(768, dtype=np.float32)  # 空嵌入用全0
            analysis_embeddings = np.expand_dims(analysis_embeddings, axis=0)
        except Exception as e:
            raise ValueError(f"CSV处理失败：{str(e)}")

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_path = os.path.join(video_frames_path[p])
                img = Image.open(img_path).convert('RGB')
                seg_imgs = [img]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        
        # 3. 特征格式处理
        extract_features=csv_data[[' pose_Rx',' pose_Ry',' gaze_0_x',' gaze_0_y',' gaze_1_x',' gaze_1_y']].values.astype(np.float32)
        signals_data_values_sl = signals_data_values_sl.astype(np.float32)
        signals_data_values_sl = np.expand_dims(signals_data_values_sl, axis=1)
        extract_features = extract_features.astype(np.float32)        
        extract_features = np.expand_dims(extract_features, axis=1)

        analysis_embeddings = analysis_embeddings.astype(np.float32)
        analysis_embeddings = np.expand_dims(analysis_embeddings, axis=1)

        return images, record.label, extract_features, signals_data_values_sl, analysis_embeddings

    def __len__(self):
        return len(self.video_list)


def train_data_loader(list_file, sl_file, text_embedding_dict, num_segments, duration, image_size, args, pre_extracted_features_path=None):
    if args.dataset == "DAiSEE":
        train_transforms = torchvision.transforms.Compose([
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    if args.dataset == "EmotiW":
        train_transforms = torchvision.transforms.Compose([
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    train_data = VideoDataset(list_file=list_file,
                              sl_file=sl_file,
                              text_embedding_dict=text_embedding_dict,
                              num_segments=num_segments,
                              duration=duration,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              pre_extracted_features_path=pre_extracted_features_path)  # 新增参数
    return train_data


def test_data_loader(list_file, sl_file, text_embedding_dict, num_segments, duration, image_size, pre_extracted_features_path=None):
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor(),
                                                     GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                     ])

    test_data = VideoDataset(list_file=list_file,
                             sl_file=sl_file,
                             text_embedding_dict=text_embedding_dict,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             pre_extracted_features_path=pre_extracted_features_path)  # 新增参数
    return test_data

