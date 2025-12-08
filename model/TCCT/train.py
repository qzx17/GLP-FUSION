import os
import time
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from data.data_processing import batch_cwt
from data.data_loader import get_source_data
from data.data_augmentation import interaug
from utilities.plotting import plot_metrics, log_metrics
from utilities.utils import EarlyStopping
from models.feature_fusion import Decision_Fusion
from test import evaluate

gpus = [0]


def train_one_epoch(model, dataloader, dataloader_sl,optimizer, scheduler, criterion_cls, 
                    original_data, original_data_sl,original_label,
                    batch_size, signal_length, device, behavioral_features, behavioral_features_sl,freq_min, freq_max,
                    tensor_height, sampling_frequency):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion_cls (nn.Module): The loss function.
        original_data (np.array): Original training data for augmentation.
        original_label (np.array): Original training labels for augmentation.
        batch_size (int): Size of the batch.
        signal_length (int): Length of the signal.
        device (torch.device): Device to run the training on (CPU or GPU).
        behavioral_features (list of str): List of behavioral features used.
        freq_min (float): Minimum frequency for CWT.
        freq_max (float): Maximum frequency for CWT.
        tensor_height (int): Number of discrete frequencies for CWT.
        sampling_frequency (int): Sampling frequency of the signal.

    Returns:
        tuple: Tuple containing epoch loss, training accuracy, epoch duration, and current learning rate.
    """
    
    # Record the start time for the epoch
    start_time = time.time()
    
    # Initialize counters for loss and accuracy calculation
    total_loss = 0
    num_batches = 0
    total_correct = 0
    total_samples = 0

    model.train()
    data_iter = iter(dataloader)
    data_iter_sl = iter(dataloader_sl)
    for _, (train_signal_data, train_label) in enumerate(tqdm(data_iter, desc="Processing")):
        # 从第二个数据加载器获取相应的批次数据
        train_signal_data_sl, train_label_sl = next(data_iter_sl)
        # 检查标签是否相等
        assert torch.equal(train_label, train_label_sl), "Labels do not match"
        # Data augmentation on the fly and concatenation with the existing batch
        aug_signal_data, aug_signal_data_sl, aug_label = interaug(
            original_data, original_data_sl, original_label, batch_size, 
            signal_length, device, num_behavioral_features=len(behavioral_features), num_behavioral_features_sl=len(behavioral_features_sl))
        
        # print(train_signal_data.shape)
        # print(train_signal_data_sl.shape)
        # print(train_label.shape)
        train_signal_data = torch.cat((train_signal_data, aug_signal_data))
        train_signal_data_sl = torch.cat((train_signal_data_sl, aug_signal_data_sl))
        train_label = torch.cat((train_label, aug_label))
        # print(train_signal_data.shape)
        # print(train_signal_data_sl.shape)
        # Apply Continuous Wavelet Transform (CWT) to the data
        frequencies = np.linspace(freq_min, freq_max, tensor_height)
        train_cwt_data = batch_cwt(train_signal_data, frequencies, sampling_frequency=sampling_frequency)
        
        # print(train_cwt_data.shape)
        
        # Forward pass through the model with both original and CWT data
        outputs = model(train_signal_data, train_cwt_data, train_signal_data_sl)
        # print(outputs)
        # print(outputs.shape)
        # print(train_label)
        # print(train_label.shape)
        loss = criterion_cls(outputs, train_label)
        total_loss += loss.item()
        num_batches += 1

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == train_label).sum().item()
        total_samples += train_label.size(0)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update learning rate scheduler and calculate metrics for the epoch
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    epoch_loss = total_loss / num_batches
    train_acc = total_correct / total_samples

    # Calculate the duration of the epoch
    end_time = time.time()
    duration = end_time - start_time

    return epoch_loss, train_acc, duration, current_lr


def train(n_classes, batch_size, b1, b2, n_epochs, lr, behavioral_features,behavioral_features_sl,
          train_folder_path, val_folder_path, test_folder_path,
          train_folder_path_sl,val_folder_path_sl,test_folder_path_sl,label_file, milestones, gamma,
          patience, sampling_frequency, weight_decay, freq_min, freq_max, tensor_height):
    """
    Train the model and evaluate it on the test set.

    Args:
        n_classes (int): Number of classes for the classification task.
        batch_size (int): Size of the batch.
        b1 (float): Beta1 hyperparameter for the Adam optimizer.
        b2 (float): Beta2 hyperparameter for the Adam optimizer.
        n_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        behavioral_features (list of str): List of behavioral features used.
        train_folder_path (str): Path to the folder containing the training data.
        test_folder_path (str): Path to the folder containing the testing data.
        label_file (str): Path to the CSV file containing the labels.
        milestones (list of int): List of epoch indices for the learning rate scheduler.
        gamma (float): Multiplicative factor of learning rate decay.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        sampling_frequency (int): Sampling frequency of the signal.
        weight_decay (float): Weight decay (L2 penalty).
        freq_min (float): Minimum frequency for CWT.
        freq_max (float): Maximum frequency for CWT.
        tensor_height (int): Number of discrete frequencies for CWT.

    Returns:
        tuple: Tuple containing the test labels and predicted labels.
    """

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the cross entropy loss and the model, and set it to the device
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    model = Decision_Fusion(n_classes)
    model = nn.DataParallel(model)
    model = model.to(device)

    # Find the length of the behavioral signal
    first_file_path = os.path.join(train_folder_path, os.listdir(train_folder_path)[0])
    signal_length = len(pd.read_csv(first_file_path))
    # Load the data from training and test directories and preserve the original for augmentation
    # print(train_folder_path_sl)
    train_signal_data, train_signal_data_sl,train_label, test_signal_data, test_signal_data_sl,test_label, val_signal_data, \
    val_signal_data_sl, val_label = get_source_data(
        train_folder_path, train_folder_path_sl,test_folder_path, test_folder_path_sl, val_folder_path, val_folder_path_sl,\
            label_file, behavioral_features,behavioral_features_sl)
    original_data = train_signal_data
    original_data_sl=train_signal_data_sl
    original_label = train_label

    # Convert the data into PyTorch tensors and transfer to the appropriate device
    train_signal_data = torch.from_numpy(train_signal_data).float().to(device)
    train_signal_data_sl = torch.from_numpy(train_signal_data_sl).float().to(device)
    train_label = torch.from_numpy(train_label).long().to(device)
    test_signal_data = torch.from_numpy(test_signal_data).float().to(device)
    test_signal_data_sl = torch.from_numpy(test_signal_data_sl).float().to(device)
    test_label = torch.from_numpy(test_label).long().to(device)
    val_signal_data = torch.from_numpy(val_signal_data).float().to(device)
    val_signal_data_sl = torch.from_numpy(val_signal_data_sl).float().to(device)
    val_label = torch.from_numpy(val_label).long().to(device)


    # Create a data loader for training data
    dataset = torch.utils.data.TensorDataset(train_signal_data, train_label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    # Create a data loader for training data
    dataset_sl = torch.utils.data.TensorDataset(train_signal_data_sl, train_label)
    dataloader_sl = torch.utils.data.DataLoader(dataset=dataset_sl, batch_size=batch_size)

    # 确保两个数据加载器的批次大小相同
    assert dataloader.batch_size == dataloader_sl.batch_size, "Batch sizes must be the same"

    # Setup the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Initialize early stopping mechanism
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Logging the size of train and val sets and the device being used
    print(f'\nTrain set size: {train_label.shape[0]}')
    print(f'Validation set size: {val_label.shape[0]}')
    print(f'\nTraining TCCT-Net...')
    print(f'Training on device: {device}')

    # Initialize lists to keep track of training and val performance
    best_acc = 0
    train_losses, train_accuracies, val_accuracies = [], [], []

    for e in range(n_epochs):
        print('\nEpoch:', e + 1)

        # Training for one epoch
        epoch_loss, train_acc, duration, current_lr = train_one_epoch(
            model, dataloader, dataloader_sl,optimizer, scheduler, criterion_cls, original_data, original_data_sl,original_label,
            batch_size, signal_length, device, behavioral_features, behavioral_features_sl,freq_min, freq_max,
            tensor_height, sampling_frequency)

        # Evaluate on test set and log metrics
        val_acc, loss_val, val_pred = evaluate(
            model, val_signal_data, val_signal_data_sl,val_label, criterion_cls, freq_min, freq_max, tensor_height, sampling_frequency)
        log_metrics(e, epoch_loss, loss_val, train_acc, val_acc, best_acc, duration, current_lr)

        # Update the best accuracy
        if val_acc > best_acc:
            best_acc = val_acc

        # Store performance metrics
        print(epoch_loss)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Check for early stopping condition
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    test_acc, loss_test, y_pred = evaluate(
        model, val_signal_data, val_signal_data_sl,val_label, criterion_cls, freq_min, freq_max, tensor_height, sampling_frequency)
    # Save the best model and plot metrics after training
    torch.save(model.module.state_dict(), '/code/TCCT_Net-main_EmotiW/model_weights.pth')
    print('The best accuracy is:', best_acc)
    # 打开一个文件用于写入，如果文件不存在则创建
    with open('/output/results.txt', 'w') as file:
        # 写入测试准确率
        file.write(f'Test Accuracy: {test_acc}\n')
        # 写入测试损失
        file.write(f'Test Loss: {loss_test}\n')
        file.write('\n')  # 写入一个换行符
    plot_metrics(train_losses, train_accuracies, val_accuracies)

    return test_label, y_pred
