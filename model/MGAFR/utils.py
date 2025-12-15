import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


## follow cpm-net's masking manner
def random_mask(view_num, input_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """

    assert missing_rate is not None
    one_rate = 1 - missing_rate

    if one_rate <= (1 / view_num): 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(input_len, view_num)) # [samplenum, viewnum] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:input_len, :]
    return matrix

def get_contrastive_loss(hidden_other,umask):
    t = hidden_other[0]
    v = hidden_other[1]
    
    t = t.contiguous().view(-1,t.shape[2])
    v = v.contiguous().view(-1,v.shape[2])
    
    umask = umask.view(-1)
    nonzero_indices = torch.nonzero(umask).view(-1)
    
    t = t[nonzero_indices]
    v = v[nonzero_indices]

    loss3 = get_contrastive_loss_one2one(t,v)
    return loss3
    
def get_contrastive_loss_one2one(z_i, z_j):
    batch_size = z_i.shape[0]
    temperature = 0.5
    mask = mask_correlated_samples(batch_size)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)

    sim = torch.matmul(z, z.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)
    
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = criterion(logits, labels)
    loss /= N

    return loss

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    mask = mask.bool()
    return mask

class Linear_Network(nn.Module):
    def __init__(self, input_size, n_classes):
        super(Linear_Network, self).__init__()
        
        self.f = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

        # self.loss = nn.MSELoss(reduction='mean')
        self.loss = nn.CrossEntropyLoss()

    def forward(self, xs, ys):
        cls_result = self.f(xs)
        loss = self.loss(cls_result, ys)
        return cls_result,loss
        
def eval_model(trainsave, data_name):

    if data_name == "DAiSEE":
        train_num = 5466
        val_num = 1704
        test_num = 1704
        epoch = 500
    elif data_name == "EmotiW":
        train_num = 5443
        val_num = 1074
        test_num = 1074
        epoch = 500

    real_save_train,real_save_val,real_save_test = {},{},{}
    
    all_hidden = trainsave["savehiddens"]
    real_save_train["savehiddens"] = all_hidden[:train_num]
    real_save_val["savehiddens"] = all_hidden[train_num:train_num+val_num]
    real_save_test["savehiddens"] = all_hidden[train_num+val_num:]

    all_label = trainsave["savelabels"]
    real_save_train["savelabels"] = all_label[:train_num]
    real_save_val["savelabels"] = all_label[train_num:train_num+val_num]
    real_save_test["savelabels"] = all_label[train_num+val_num:]

    all_hidden = torch.tensor(all_hidden).cuda()
    all_label = torch.tensor(all_label).cuda()

    H_train = all_hidden[:train_num]
    train_labels = all_label[:train_num]
    H_val = all_hidden[train_num:train_num+val_num]
    val_labels = all_label[train_num:train_num+val_num]
    H_test = all_hidden[train_num+val_num:]
    test_labels = all_label[train_num+val_num:]

    linear_batch = 2048
    train_dataset = TensorDataset(H_train,train_labels)
    linear_train_loader = DataLoader(dataset=train_dataset, batch_size=linear_batch, shuffle=False)
    val_dataset = TensorDataset(H_val,val_labels)
    linear_val_loader = DataLoader(dataset=val_dataset, batch_size=linear_batch, shuffle=False)
    test_dataset = TensorDataset(H_test,test_labels)
    linear_test_loader = DataLoader(dataset=test_dataset, batch_size=linear_batch, shuffle=False)
    
    linear_model = Linear_Network(H_train.shape[1],4).cuda()
    linear_optimizer = optim.Adam(linear_model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    old_metrics = {}
    old_metrics['Top-1 Accuracy'] = 0

    for epoch in range(epoch):
        train_results,train_preds = train_or_eval_linear(linear_model, linear_train_loader, optimizer=linear_optimizer, train=True)
        val_results,val_preds = train_or_eval_linear(linear_model, linear_val_loader, optimizer=None, train=False)
        test_results,test_preds = train_or_eval_linear(linear_model, linear_test_loader, optimizer=None, train=False)
        
        if old_metrics['Top-1 Accuracy'] < val_results['Top-1 Accuracy']:
            old_metrics = test_results
        print(test_results)
            
    return old_metrics, {"train":real_save_train, "val":real_save_val, "test":real_save_test}

def train_or_eval_linear(model, dataloader, optimizer=None, train=True):
    cuda = torch.cuda.is_available()
    preds, labels = [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for xs, ys in dataloader:
        if train: optimizer.zero_grad()
        
        xs = xs.to(torch.float32)
        ys = ys.to(torch.int64)
        
        pred, loss = model(xs, ys)

        preds.append(pred.data.cpu().numpy())
        labels.append(ys.data.cpu().numpy())
        
        if train:
            loss.backward()
            optimizer.step()
        
    preds = np.concatenate(preds,axis=0)
    labels = np.concatenate(labels,axis=0)
    
    top1_preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, top1_preds)
    f1 = f1_score(labels, top1_preds, average='weighted')
    mae = mean_absolute_error(labels, top1_preds)
    eval_results = {
        "Top-1 Accuracy": round(accuracy, 4),
        "F1-Score": round(f1, 4),
        "MAE": round(mae, 4)
    }
        
    return eval_results,preds


class MaskedReconLoss(nn.Module):

    def __init__(self):
        super(MaskedReconLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, recon_input, target_input, input_mask, umask, tdim, vdim):
        """ ? => refer to spk and modality
        recon_input  -> ? * [seqlen, batch, dim]
        target_input -> ? * [seqlen, batch, dim]
        input_mask   -> ? * [seqlen, batch, dim]
        umask        -> [batch, seqlen]
        """
        assert len(recon_input) == 1
        recon = recon_input[0] # [seqlen, batch, dim]
        target = target_input[0] # [seqlen, batch, dim]
        mask = input_mask[0] # [seqlen, batch, 2]
        
        recon  = torch.reshape(recon, (-1, recon.size(2)))   # [seqlen*batch, dim]
        target = torch.reshape(target, (-1, target.size(2))) # [seqlen*batch, dim]
        mask   = torch.reshape(mask, (-1, mask.size(2)))     # [seqlen*batch, 2] 1(exist); 0(mask)
        umask = torch.reshape(umask.permute(1,0), (-1, 1)) # [seqlen*batch, 1]

        L_rec = recon[:, :tdim]
        V_rec = recon[:, tdim:]
        L_full = target[:, :tdim]
        V_full = target[:, tdim:]

        L_miss_index = torch.reshape(mask[:, 0], (-1, 1))
        V_miss_index = torch.reshape(mask[:, 1], (-1, 1))
        
        loss_recon2 = self.loss(L_rec*umask, L_full*umask) * L_miss_index
        loss_recon3 = self.loss(V_rec*umask, V_full*umask) * V_miss_index

        loss_recon2 = torch.sum(loss_recon2) / tdim
        loss_recon3 = torch.sum(loss_recon3) / vdim
        loss_recon = (loss_recon2 + loss_recon3) / torch.sum(umask)

        return loss_recon
    
