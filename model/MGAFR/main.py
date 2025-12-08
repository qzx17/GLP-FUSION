import os
import time
import argparse
from torch.optim.adam import Adam
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from dataloader.tensor_dataloader import train_data_loader, test_data_loader
from mgafr import MGAFR
from utils import MaskedReconLoss
from utils import get_contrastive_loss
from utils import random_mask
from utils import eval_model
from tqdm import tqdm

def train_MGAFR(args, model, rec_loss_fn, dataloader, vdim, tdim, mask_rate=None, optimizer=None, train=False):
    
    # --- 移植自目标代码：初始化所有需要的列表 ---
    savelabels, savehiddens = [], []
    losses, losses1, losses2, losses3 = [], [], [], []
    labels, masks = [], []
    
    save_raw_video, save_raw_text = [], []
    save_recon_video, save_recon_text = [], []
    # -------------------------------------------

    cuda = torch.cuda.is_available()

    assert not train or optimizer is not None
    if train:
        model.train()
    else:
        model.eval()
        
    for data in tqdm(dataloader, desc="Training" if train else "Validation"):
        
        if train: optimizer.zero_grad()
        
        # --- 你的数据加载和预处理部分 (保持不变) ---
        visual_features, batch_labels, _, _, text_features = data
        
        visual_features = visual_features.cuda()
        text_features = text_features.cuda()
        batch_labels = batch_labels.cuda()
        
        visual_features = visual_features.squeeze(0).squeeze(1).transpose(0, 1)
        text_features = text_features.squeeze(1).squeeze(1).transpose(0, 1)
        seqlen, batch = visual_features.size(0), visual_features.size(1)


        umask = torch.ones(batch, seqlen, device=visual_features.device)
        
        view_num = 2
        matrix = random_mask(view_num, seqlen*batch, mask_rate)
        text_mask = torch.from_numpy(np.reshape(matrix[:, 0], (seqlen, batch, 1))).cuda()
        visual_mask = torch.from_numpy(np.reshape(matrix[:, 1], (seqlen, batch, 1))).cuda()
        
        masked_text = text_features * text_mask
        masked_visual = visual_features * visual_mask
        
        input_features_tensor = torch.cat([text_features, visual_features], dim=2)
        masked_input_features_tensor = torch.cat([masked_text, masked_visual], dim=2)
        input_features_mask_tensor = torch.cat([text_mask, visual_mask], dim=2)
        
        # --- 为了和模型输出结构对齐，加上一个维度 ---
        input_features = input_features_tensor.unsqueeze(0)
        masked_input_features = masked_input_features_tensor.unsqueeze(0)
        input_features_mask = input_features_mask_tensor.unsqueeze(0)
        # -------------------------------------------

        # --- 适配目标代码的label形状 [B] -> [B, S] ---
        label = batch_labels.unsqueeze(1).repeat(1, seqlen)

        with torch.set_grad_enabled(train):
            recon_input_features, hidden, hidden_other = model(masked_input_features, umask, input_features_mask)
            
            # --- 移植自目标代码：一次性、更健壮的损失计算 ---
            contrastive_loss = get_contrastive_loss(hidden_other, umask)
            # 注意：这里的 rec_loss_fn 应该就是你定义的 rec_loss
            reconstruct_loss = rec_loss_fn(recon_input_features, input_features, input_features_mask, umask, tdim, vdim) * args.recon_weight
            featureInfo_loss = hidden_other[2] * args.inf_weight
            
            loss = contrastive_loss + reconstruct_loss + featureInfo_loss
            # ------------------------------------------------

        # --- 反向传播 (保持不变) ---
        if train:
            loss.backward()
            optimizer.step()
        
        # --- 移植自目标代码：详细的结果保存逻辑 ---
        tempseqlen = np.sum(umask.cpu().data.numpy(), 1)
        temphidden = hidden.transpose(0,1).cpu().data.numpy()
        templabel = label.cpu().data.numpy()

        temp_raw_text = text_features.transpose(0,1).cpu().data.numpy()
        temp_raw_video = visual_features.transpose(0,1).cpu().data.numpy()
        temp_recon_text = recon_input_features[0][:,:,0:tdim].transpose(0,1).cpu().data.numpy() # <--- 修改索引
        temp_recon_video = recon_input_features[0][:,:,tdim:].transpose(0,1).cpu().data.numpy() # <--- 修改索引
        
        for ii in range(len(tempseqlen)):
            xii = int(tempseqlen[ii])
            if xii == 0: continue # 跳过完全是padding的样本
            
            itemhidden = temphidden[ii][:xii, :]
            itemlabel  = templabel[ii][:xii]
            savehiddens.append(itemhidden)
            savelabels.append(itemlabel)
            
            item_raw_text = temp_raw_text[ii][:xii, :]
            item_raw_video = temp_raw_video[ii][:xii, :]
            save_raw_text.append(item_raw_text)
            save_raw_video.append(item_raw_video)
            item_recon_text = temp_recon_text[ii][:xii, :]
            item_recon_video = temp_recon_video[ii][:xii, :]
            save_recon_text.append(item_recon_text)
            save_recon_video.append(item_recon_video)

        # --- 移植自目标代码：标准的损失聚合方式 ---
        labels.append(label.reshape(-1).data.cpu().numpy())
        masks.append(umask.reshape(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        losses1.append(contrastive_loss.item()*masks[-1].sum())
        losses2.append(reconstruct_loss.item()*masks[-1].sum())
        losses3.append(featureInfo_loss.item()*masks[-1].sum())
        # -----------------------------------------

    # --- 移植自目标代码：标准的平均损失计算和最终结果打包 ---
    masks  = np.concatenate(masks)
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_loss1 = round(np.sum(losses1)/np.sum(masks), 4)
    avg_loss2 = round(np.sum(losses2)/np.sum(masks), 4)
    avg_loss3 = round(np.sum(losses3)/np.sum(masks), 4)
        
    save_dict = {}
    if len(savehiddens) > 0:
        save_dict["save_raw_text"] = np.concatenate(save_raw_text,axis=0)
        save_dict["save_raw_video"] = np.concatenate(save_raw_video,axis=0)
        save_dict["save_recon_text"] = np.concatenate(save_recon_text,axis=0)
        save_dict["save_recon_video"] = np.concatenate(save_recon_video,axis=0)
        save_dict["savehiddens"] = np.concatenate(savehiddens,axis=0)
        save_dict["savelabels"] = np.concatenate(savelabels,axis=0)
    # ----------------------------------------------------

    return [avg_loss, avg_loss1, avg_loss2, avg_loss3], save_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # MODIFIED: 简化了参数，使其与您的项目更相关
    parser.add_argument('--dataset', type=str, default='DAiSEE', help='dataset type (DAiSEE or EmotiW)')
    # parser.add_argument('--dataset', type=str, default='EmotiW', help='dataset type (DAiSEE or EmotiW)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=70, metavar='E', help='number of epochs')
    parser.add_argument('--mask-rate', type=float, default=0.0, help='mask rate for modalities')
    parser.add_argument('--recon_weight', type=float, default=10.0, help='recon loss weight')
    parser.add_argument('--inf_weight', type=float, default=1.0, help='informative loss weight')
    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()

    # --- MODIFIED: 使用您的数据加载器 ---
    print (f'====== Reading Data for {args.dataset} =======')
    # 定义您的文件路径
    TRAIN_LIST_FILE = '/root/autodl-tmp/code/Student_Engagement/DAiSEE_all.txt'
    # TRAIN_LIST_FILE = '/root/autodl-tmp/code/Student_Engagement/EmotiW_all.txt'
    # TRAIN_LIST_FILE = '/root/autodl-tmp/code/Student_Engagement/DAiSEE_Test_image.txt'
    TRAIN_TEXT_FILE = '/root/autodl-tmp/code/Student_Engagement/embedding_dict_timeseries_daisee_16.pkl'
    # TRAIN_TEXT_FILE = '/root/autodl-tmp/code/Student_Engagement/embedding_dict_timeseries_emotiw_16.pkl'
    SL_FILE = '/root/autodl-tmp/code/Student_Engagement/All_daisee.csv'
    # SL_FILE = '/root/autodl-tmp/code/Student_Engagement/EmotiW_Train.csv'
    TRAIN_VISUAL_FEATURES = '/root/autodl-tmp/code/Student_Engagement/preprocessed_features/train_visual_features.pkl'
    # TRAIN_VISUAL_FEATURES = '/root/autodl-tmp/code/Student_Engagement/preprocessed_features/train_visual_features_emotiw.pkl'

    train_dataset = train_data_loader(
        list_file=TRAIN_LIST_FILE, sl_file=SL_FILE, text_embedding_dict=TRAIN_TEXT_FILE,
        num_segments=16, duration=1, image_size=112, args=args, pre_extracted_features_path=TRAIN_VISUAL_FEATURES
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 定义特征维度
    adim = 0      # 没有音频
    tdim = 768    # 文本特征维度
    vdim = 2048   # 视觉特征维度
    # -----------------------------------------------

    print (f'====== Training and Evaluation =======')
    print (f'Step1: build model')
    model = MGAFR(tdim, vdim, n_classes=1) # n_classes在预训练中不重要
    rec_loss = MaskedReconLoss()

    if cuda:
        model.cuda()
        rec_loss.cuda()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print (f'Step2: training')
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        train_loss, trainsave = train_MGAFR(args, model, rec_loss, train_loader, vdim, tdim, mask_rate=args.mask_rate, optimizer=optimizer, train=True)
        print(f'Epoch:{epoch+1}; Train Loss:{train_loss[0]:.4f}; Recon:{train_loss[2]:.4f}; Contrast:{train_loss[1]:.4f}; Info:{train_loss[3]:.4f}')

        if best_loss > train_loss[0]:
            best_loss = train_loss[0]
            best_trainsave = trainsave
    
    test_results, test_save_result = eval_model(best_trainsave, args.dataset)

    print (f'====== Saving =======')    
    print(test_results)
    log_result_file = f'./logs/{args.dataset.lower()}_result.txt'
    with open(log_result_file, 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'\n')
        f.write(str(test_results)+'\n')
        f.write('\n')

    print(f'====== Finish =======')