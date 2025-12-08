import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_csv_data(folder_path, folder_path_sl,label_file, behavioral_features,behavioral_features_sl):
    """
    Load data from CSV files in a folder and corresponding labels.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Array of data extracted from the CSV files.
        np.array: Array of labels corresponding to the data.
    """
        
    labels_df = pd.read_csv(label_file)
    all_data_pose, all_labels, all_data_sl= [], [] , []
    folder_path_sl = pd.read_csv(folder_path_sl)
    i=0
    # Process each CSV file in the directory
    for filename in tqdm(os.listdir(folder_path), desc="Loading data"):
        if filename.endswith('.csv'):
            subject_id = filename.split('.')[0]
            subject_file = os.path.join(folder_path, filename)
            subject_data = pd.read_csv(subject_file)
            if (subject_id + '.mp4') in folder_path_sl['video_names'].values or (subject_id + '.avi') in folder_path_sl['video_names'].values:
                # print(subject_id)
                matching_rows = folder_path_sl[(folder_path_sl['video_names'] == subject_id + '.mp4') | (folder_path_sl['video_names'] == subject_id + '.avi')]
            signals_data_values_sl = matching_rows[behavioral_features_sl].values
            # 找到包含 NaN 的行的索引
            nan_rows_indices = np.where(np.isnan(signals_data_values_sl).any(axis=1))[0]            
            # 检查是否有 NaN 值
            if len(nan_rows_indices) > 0:
                print(f"subject_id {subject_id} 中包含 NaN 的行有：")
                for index in nan_rows_indices:
                    print(f"行号 {index}:")
                    print(signals_data_values_sl[index])
                continue
            # 找到包含无穷大值的行的索引
            inf_rows_indices = np.where(np.isinf(signals_data_values_sl).any(axis=1))[0]
            # 检查是否有无穷大值
            if len(inf_rows_indices) > 0:
                print(f"subject_id {subject_id} 中包含无穷大值的行有：")
                for index in inf_rows_indices:
                    print(f"行号 {index}:")
                    print(signals_data_values_sl.loc[index])
                continue
            # Stack data for selected behavioral features
            subject_data_values = np.stack([subject_data[col].values for col in behavioral_features], axis=0)
            subject_label = labels_df[labels_df['chunk'].str.contains(subject_id)]['label'].values
            # print(subject_data_values.dtype)
            # print(signals_data_values_sl.dtype)
            if signals_data_values_sl.dtype != np.float64:
                # 如果数据类型不是 float64，打印信息并跳过当前循环迭代
                print(subject_id)
                i+=1
            # Append data and label if label exists
            if len(subject_label) > 0:
                all_data_pose.append(subject_data_values)
                all_labels.append(subject_label[0])
                all_data_sl.append(signals_data_values_sl)
                if np.array(all_data_sl).dtype == object:
                    all_data_pose.pop()  # 移除最后一个元素
                    all_labels.pop()  # 移除最后一个元素
                    all_data_sl.pop()  # 移除最后一个元素
                # print(all_data_pose.dtype)
                # print(subject_label[0])
                # print(all_data_sl.dtype)
            else:
                print(f"No label found for subject {subject_id}")
    print(i)
    # Reshape the collected data and convert it to numpy arrays
    all_data_pose = np.array(all_data_pose)
    print(all_data_pose.dtype)
    all_data_sl=np.array(all_data_sl)
    all_labels = np.array(all_labels)
    all_data_pose = np.expand_dims(all_data_pose, axis=1)  
    # print(all_data_pose)
    # print(all_data_sl)
    return all_data_pose, all_data_sl,all_labels


def get_source_data(train_folder_path, train_folder_path_sl,val_folder_path, val_folder_path_sl, test_folder_path, test_folder_path_sl,\
            label_file, behavioral_features,behavioral_features_sl):
    """
    Load and preprocess training and testing data from the specified folders.

    Args:
        train_folder_path (str): Path to the folder containing the training data.
        test_folder_path (str): Path to the folder containing the testing data.
        label_file (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.

    Returns:
        np.array: Processed training data.
        np.array: Labels for the training data.
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load training data
    print('\nLoading train data ...')
    train_data, train_data_sl, train_labels = load_csv_data(train_folder_path, train_folder_path_sl,label_file, behavioral_features,behavioral_features_sl)
    train_labels = train_labels.reshape(1, -1)

    # Shuffle the training data
    shuffle_index = np.random.permutation(len(train_data))
    # train_data = train_data[shuffle_index, :,:,:]
    train_data = train_data[shuffle_index, :]
    print(train_data_sl.shape)
    train_data_sl=train_data_sl[shuffle_index, :]
    train_labels = train_labels[0][shuffle_index]
    if np.isnan(train_data_sl).any():
        print("原始数据中包含 NaN")

        # 找到包含 NaN 的行的索引
        nan_rows_indices = np.where(np.isnan(train_data_sl).any(axis=(1, 2)))[0]  # 检查每一行是否有 NaN
        rows_with_nan = train_data_sl[nan_rows_indices]  # 筛选出包含 NaN 的行

        print("包含 NaN 的行有：")
        for index, row in enumerate(nan_rows_indices):
            print(f"行号 {row}:")
            print(rows_with_nan[index])
   
    # print(train_data.shape)
    # print(train_data_sl.shape)
    # Standardize both train and test data using training data statistics
    target_mean = np.mean(train_data)
    
    # train_data_sl=np.log(train_data_sl[:, :, :2])
    # test_data_sl = np.log(test_data_sl[:, :, :2])
    # val_data_sl=np.log(val_data_sl[:, :, :2])
    # 计算平均值
    target_mean_sl = np.mean(train_data_sl)
    print(train_data_sl.shape)
    target_mean_sl=np.mean(train_data_sl, axis=(0,1),keepdims=True)
    print(target_mean_sl)
    target_std = np.std(train_data)
    target_std_sl=np.std(train_data_sl, axis=(0,1), keepdims=True)

     # Load validating data
    print('\nLoading val data ...')
    val_data, val_data_sl,val_labels = load_csv_data(val_folder_path, val_folder_path_sl,label_file, behavioral_features,behavioral_features_sl)
    val_labels = val_labels.reshape(-1)  

     # Load testing data
    print('\nLoading test data ...')
    test_data, test_data_sl,test_labels = load_csv_data(test_folder_path, test_folder_path_sl,label_file, behavioral_features,behavioral_features_sl)
    test_labels = test_labels.reshape(-1)  
    train_data = (train_data - target_mean) / target_std
    train_data_sl=(train_data_sl-target_mean_sl)/target_std_sl
    # 检查计算平均值后是否包含 NaN
    if np.isnan(target_mean_sl.any()):
        print("计算平均值后出现了 NaN")
    test_data = (test_data - target_mean) / target_std    
    test_data_sl=(test_data_sl-target_mean_sl)/target_std_sl
    val_data = (val_data-target_mean) / target_std
    val_data_sl = (val_data_sl-target_mean_sl) / target_std_sl

    return train_data, train_data_sl,train_labels, test_data, test_data_sl,test_labels, val_data, val_data_sl, val_labels


def get_source_data_inference(inference_folder_path, label_file_inference,
                              behavioral_features, target_mean, target_std):
    """
    Load and preprocess inference data from the specified folder.

    Args:
        inference_folder_path (str): Path to the folder containing the inference data.
        label_file_inference (str): Path to the CSV file containing the labels.
        behavioral_features (list of str): List of behavioral features to extract from the data.
        target_mean (float): Mean value of the training data used for standardization.
        target_std (float): Standard deviation of the training data used for standardization.

    Returns:
        np.array: Processed testing data.
        np.array: Labels for the testing data.
    """

    # Load inference data
    print('\nLoading data for inference ...')
    inference_data, inference_labels = load_csv_data(inference_folder_path, label_file_inference, behavioral_features)
    inference_labels = inference_labels.reshape(-1)

    # Standardize inference data using provided training data statistics
    inference_data = (inference_data - target_mean) / target_std

    return inference_data, inference_labels
