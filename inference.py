import torch
import torch.nn as nn
from model.GLP_fusion.feature_fusion import Decision_Fusion
# from dataloader.video_new import test_data_loader
from dataloader.video_compare import test_data_loader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import os
rmpy as np
# test_annotation_file_path = "/DAiSEE_Test_set.txt"
# sl_file_path = "SL_DAiSEE.csv"

#防止出现在val上高，在test低的极端现象，采用平均的方法，验证多次
MODEL_PATHS = [
    "/pth/EmotiW/model1.pth",
    "/pth/EmotiW/model2.pth",
    "/pth/EmotiW/model3.pth",
]
RESULT_DIR = "result"
test_annotation_file_path = "/EmotiW_Test_set.txt"
sl_file_path = "SL_EmotiW.csv"
def load_model(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model = Decision_Fusion(n_classes=4)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    model = model.cuda()
    model.eval() 
    
    return model

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Not-Engaged', 'Barely-Engaged', 'Engaged', 'Highly-Engaged']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def inference(test_loader, model):
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for i, (images, target, extract_features, extract_features_sl) in enumerate(test_loader):
            images = images.cuda()
            extract_features = extract_features.cuda()
            extract_features_sl = extract_features_sl.cuda()
            
            output = model(images, extract_features, extract_features_sl)
            probabilities = torch.softmax(output, dim=1) 
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, 
        all_predictions, 
        average='weighted'
    )

    correlation = np.corrcoef(all_targets, all_predictions)[0, 1]

    mae = np.mean(np.abs(all_targets - all_predictions))
    
    cm = confusion_matrix(all_targets, all_predictions)
    plot_confusion_matrix(cm, 'result/confusion_matrix.png')
    
    print(f'Weighted Accuracy: {accuracy:.4f}')
    print(f'Weighted Precision: {precision:.4f}')
    print(f'Weighted Recall: {recall:.4f}')
    print(f'Weighted F1 Score: {f1:.4f}')
    print(f'Correlation Coefficient: {correlation:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')
    
    with open('/result/evaluation_results.txt', 'w') as f:
        f.write(f'Weighted Accuracy: {accuracy:.4f}\n')
        f.write(f'Weighted Precision: {precision:.4f}\n')
        f.write(f'Weighted Recall: {recall:.4f}\n')
        f.write(f'Weighted F1 Score: {f1:.4f}\n')
        f.write(f'Correlation Coefficient: {correlation:.4f}\n')
        f.write(f'Mean Absolute Error: {mae:.4f}\n')
        
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, 
            all_predictions
        )
        classes = ['Not-Engaged', 'Barely-Engaged', 'Engaged', 'Highly-Engaged']
        f.write('\nPer-class metrics:\n')
        for i, class_name in enumerate(classes):
            f.write(f'\n{class_name}:\n')
            f.write(f'Precision: {precision_per_class[i]:.4f}\n')
            f.write(f'Recall: {recall_per_class[i]:.4f}\n')
            f.write(f'F1-score: {f1_per_class[i]:.4f}\n')
    
    return accuracy, precision, recall, f1, correlation, mae

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    test_loader = test_data_loader(list_file=test_annotation_file_path,
                                  sl_file=sl_file_path,
                                  num_segments=16,
                                  duration=1,
                                  image_size=112)
    test_loader = torch.utils.data.DataLoader(test_loader,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=8,
                                             pin_memory=True)

    metrics_list = []
    
    for idx, model_path in enumerate(MODEL_PATHS):
        model = load_model(model_path)
        metrics = inference(test_loader, model)
        metrics_list.append(metrics)
    
    metrics_array = np.array(metrics_list)  # shape: (num_models, 6)
    metrics_names = [
        "Accuracy", "Precision", "Recall", 
        "F1 Score", "Correlation", "MAE"
    ]
    metrics_mean = np.mean(metrics_array, axis=0)
    metrics_std = np.std(metrics_array, axis=0)
    summary_path = os.path.join(RESULT_DIR, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Multi-Model Evaluation Summary\n")
        f.write("===============================\n")
        f.write(f"Number of models: {len(MODEL_PATHS)}\n\n")
        f.write("Individual Model Results:\n")
        for idx, metrics in enumerate(metrics_list):
            f.write(f"\nModel {idx}:\n")
            for name, value in zip(metrics_names, metrics):
                f.write(f"{name}: {value:.4f}\n")
        f.write("\n\nStatistical Results (Mean ± Std):\n")
        f.write("==================================\n")
        for name, mean, std in zip(metrics_names, metrics_mean, metrics_std):
            f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")
    
    # 8. 打印汇总结果
    print("\n\n=====================================")
    print("Multi-Model Evaluation Summary")
    print("=====================================")
    print(f"Number of models: {len(MODEL_PATHS)}")
    print("\nStatistical Results (Mean ± Std):")
    for name, mean, std in zip(metrics_names, metrics_mean, metrics_std):
        print(f"{name}: {mean:.4f} ± {std:.4f}")
    print("=====================================")


class RecorderMeter(object):
    
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


if __name__ == '__main__':
    main()
