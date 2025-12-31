import matplotlib.pyplot as plt
import os

def plot_metrics(train_losses, train_accuracies, val_accuracies):
    """
    Plot training loss, training accuracy, and test accuracy over epochs.

    Args:
        train_losses (list of float): List of training losses over epochs.
        train_accuracies (list of float): List of training accuracies over epochs.
        test_accuracies (list of float): List of test accuracies over epochs.
    """
    
    # Determine the number of epochs 
    epochs = range(1, len(train_losses) + 1)

    # Create a plot for training loss 
    fig, ax1 = plt.subplots(dpi=300)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a secondary axis to plot training accuracy
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Train Accuracy', color=color)  
    ax2.plot(epochs, train_accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    fig.savefig('/code/SL_TCCT_Net-main/train_metrics.png', bbox_inches='tight')
    print('\nPlot of training loss and training accuracy saved to train_metrics.png')

    # Plot validating accuracy 
    fig, ax = plt.subplots(dpi=300)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.plot(epochs, val_accuracies, label='Validation Accuracy')
    ax.legend()

    fig.tight_layout()
    fig.savefig('/output/val_accuracy.png', bbox_inches='tight')
    print('Plot of val accuracy saved to val_accuracy.png')


def log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, best_acc, duration, learning_rate):
    """
    Log training and testing metrics to the console.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the epoch.
        test_loss (float): Testing loss for the epoch.
        train_acc (float): Training accuracy for the epoch.
        test_acc (float): Testing accuracy for the epoch.
        best_acc (float): Best observed testing accuracy.
        duration (float): Duration of the epoch in seconds.
        learning_rate (float): Current learning rate.
    """
    print(f'Epoch: {epoch + 1}',
          f'Train Loss: {train_loss:.4f}',
          f'Validation Loss: {val_loss:.4f}',
          f'Train Acc: {train_acc:.4f}',
          f'Validation Acc: {val_acc:.4f}',
          f'Best Acc: {best_acc:.4f}',
          f'Time: {duration:.2f}s',
          f'LR: {learning_rate:.6f}', sep='  |  ')
    # 定义输出文件夹和文件名
    output_folder_1 = '/output'
    output_folder_2 = '/code/SL_TCCT_Net-main/output'
    file_name = 'result_train_val.txt'
    output_path_1 = os.path.join(output_folder_1, file_name)
    output_path_2 = os.path.join(output_folder_2, file_name)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder_1):
        os.makedirs(output_folder_1)
    if not os.path.exists(output_folder_2):
        os.makedirs(output_folder_2)

    # 准备要输出的字符串
    output_str = f'Epoch: {epoch + 1} '
    output_str += f'Train Loss: {train_loss:.4f} '
    output_str += f'Validation Loss: {val_loss:.4f} '
    output_str += f'Train Acc: {train_acc:.4f} '
    output_str += f'Validation Acc: {val_acc:.4f} '
    output_str += f'Best Acc: {best_acc:.4f} '
    output_str += f'Time: {duration:.2f}s '
    output_str += f'LR: {learning_rate:.6f}'

    # 将输出写入文件
    with open(output_path_1, 'a') as f:  # 使用 'a' 模式以追加方式写入
        f.write(output_str + '\n')  # 在字符串末尾添加换行符
    with open(output_path_2, 'a') as f:  # 使用 'a' 模式以追加方式写入
        f.write(output_str + '\n')  # 在字符串末尾添加换行符