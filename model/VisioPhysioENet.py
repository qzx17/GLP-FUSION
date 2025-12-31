import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataloader.video_sl import train_data_loader, test_data_loader
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=48)

parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr-image-encoder', type=float, default=1e-5)
parser.add_argument('--lr-prompt-learner', type=float, default=1e-3)

parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int)

parser.add_argument('--contexts-number', type=int, default=4)
parser.add_argument('--class-token-position', type=str, default="end")
parser.add_argument('--class-specific-contexts', type=str, default='False')
parser.add_argument('--load_and_tune_prompt_learner', type=str, default='False', help='Whether to load and tune the prompt learner')
parser.add_argument('--text-type', type=str)
parser.add_argument('--exper-name', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--temporal-layers', type=int, default=1)

args = parser.parse_args()
def get_data_from_loader(data_loader):
    """从数据加载器中提取数据并分离眼动和生理特征"""
    all_eye_features = []
    all_physio_features = []
    all_labels = []
    
    for _, target, _, extract_features_sl in data_loader:
        features = extract_features_sl.numpy()
        # 重塑数据形状
        features = features.reshape(features.shape[0], -1)  # 将(batch_size, 1, 1, 7)转换为(batch_size, 7)
        
        # 分离眼动特征和生理特征
        eye_features = features[:, :3]  # 前3列是眼动特征
        physio_features = features[:, 3:7]  # 后4列是生理特征
        
        all_eye_features.append(eye_features)
        all_physio_features.append(physio_features)
        all_labels.append(target.numpy())
    
    # 合并所有batch的数据
    X_eye = np.concatenate(all_eye_features, axis=0)
    X_physio = np.concatenate(all_physio_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # 打印数据形状以便调试
    print(f"Reshaped eye features shape: {X_eye.shape}")
    print(f"Reshaped physio features shape: {X_physio.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X_eye, X_physio, y

def main(set):
    # 只保留数据加载相关的代码
    if args.dataset=="DAiSEE":
        train_annotation_file_path = "DAiSEE_Train_set.txt"
        test_annotation_file_path = "DAiSEE_Test_set.txt"
        sl_file_path = "SL_DAiSEE.csv"
    elif args.dataset=="EngageNet":
        train_annotation_file_path = "EngageNet_Train_set.txt"
        test_annotation_file_path = "EngageNet_Test_set.txt"
        sl_file_path = "SL_EngageNet.csv"

    # 加载数据
    train_data = train_data_loader(
        list_file=train_annotation_file_path,
        sl_file=sl_file_path,
        num_segments=16,
        duration=1,
        image_size=112,
        args=args
    )
    
    test_data = test_data_loader(
        list_file=test_annotation_file_path,
        sl_file=sl_file_path,
        num_segments=16,
        duration=1,
        image_size=112
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # 获取训练和测试数据
    X_eye_train, X_physio_train, y_train = get_data_from_loader(train_loader)
    X_eye_test, X_physio_test, y_test = get_data_from_loader(test_loader)
    
    # 预处理
    imputer = SimpleImputer(strategy='median')
    X_eye_train = imputer.fit_transform(X_eye_train)
    X_eye_test = imputer.transform(X_eye_test)
    
    X_physio_train = imputer.fit_transform(X_physio_train)
    X_physio_test = imputer.transform(X_physio_test)
    
    # 标准化生理特征
    scaler = skl.preprocessing.StandardScaler().fit(X_physio_train)
    X_physio_train = scaler.fit_transform(X_physio_train)
    X_physio_test = scaler.transform(X_physio_test)
    
    # 训练眼动模型
    visual_model = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
    visual_model.fit(X_eye_train, y_train)
    
    # 训练生理模型
    physio_model = RandomForestClassifier(n_estimators=100, random_state=42)
    physio_model.fit(X_physio_train, y_train)
    
    # 获取预测概率
    visual_preds_train = visual_model.predict_proba(X_eye_train)
    visual_preds_test = visual_model.predict_proba(X_eye_test)
    physio_preds_train = physio_model.predict_proba(X_physio_train)
    physio_preds_test = physio_model.predict_proba(X_physio_test)
    
    # 尝试不同的权重组合
    visual_weights = np.arange(0, 1.1, 0.1)  # 0到1，步长0.1
    accuracies = []
    
    for visual_weight in visual_weights:
        physio_weight = 1 - visual_weight
        
        # 特征融合
        fused_train = np.concatenate([
            visual_preds_train * visual_weight,
            physio_preds_train * physio_weight
        ], axis=1)
        
        fused_test = np.concatenate([
            visual_preds_test * visual_weight,
            physio_preds_test * physio_weight
        ], axis=1)
        
        # 元分类器
        meta_classifier = LogisticRegression(solver='lbfgs', random_state=42)
        meta_classifier.fit(fused_train, y_train)
        
        # 预测和评估
        final_predictions = meta_classifier.predict(fused_test)
        accuracy = accuracy_score(y_test, final_predictions)
        accuracies.append(accuracy)
        
        print(f"Weight combination - Visual: {visual_weight:.1f}, Physio: {physio_weight:.1f}")
        print(f"Fusion accuracy: {accuracy:.4f}")
    
    # 找到最佳权重组合
    best_idx = np.argmax(accuracies)
    best_visual_weight = visual_weights[best_idx]
    best_accuracy = accuracies[best_idx]
    
    print("\nBest Results:")
    print(f"Best visual weight: {best_visual_weight:.2f}")
    print(f"Best physio weight: {(1-best_visual_weight):.2f}")
    print(f"Best fusion accuracy: {best_accuracy:.4f}")
    
    # 绘制权重-准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(visual_weights, accuracies, marker='o')
    plt.xlabel('Visual Weight')
    plt.ylabel('Accuracy')
    plt.title('Fusion Performance vs Visual Weight')
    plt.grid(True)
    plt.show()
    
    # 打印单模型性能作为对比
    print("\nSingle Model Performance:")
    print(f"Visual model accuracy: {accuracy_score(y_test, visual_model.predict(X_eye_test)):.4f}")
    print(f"Physio model accuracy: {accuracy_score(y_test, physio_model.predict(X_physio_test)):.4f}")
    
    return best_visual_weight, best_accuracy

if __name__ == '__main__':
    best_weight, best_acc = main(0)
    print("best_weight: ", best_weight)
    print("best_accuracy: ", best_acc)

   