import torch
import torch.nn as nn

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x
    
class MGAFR(nn.Module):
    def __init__(self,tdim, vdim, n_classes):
        super(MGAFR, self).__init__()
        
        self.tdim = tdim
        self.vdim = vdim
        
        self.encode_dim = 2048
    
        self.Wt = nn.Sequential(
            nn.Linear(tdim, self.encode_dim),
        )
        self.Wv = nn.Sequential(
            nn.Linear(vdim, self.encode_dim),
        )

        self.decoder_text = nn.Sequential(
            nn.Linear(self.encode_dim, tdim),
        )
        self.decoder_video = nn.Sequential(
            nn.Linear(self.encode_dim, vdim),
        )
        
        self.normalization = "NormAdj"
        self.degree = 1
        self.alpha = 0.75 
        self.k = 4
        
        self.k1 = 1
        self.k2 = 1
        self.mu = 0.5 

        self.weight_t = LinearLayer(self.encode_dim, self.encode_dim)
        self.weight_v = LinearLayer(self.encode_dim, self.encode_dim)
        
    def aug_normalized_adjacency(self,adj):
        adj = adj + torch.eye(adj.shape[0]).cuda()
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.where(row_sum != 0, 1.0 / torch.sqrt(row_sum), torch.zeros(1).cuda())
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt).cuda()
        adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return adj

    def normalized_adjacency(self,adj):
        row_sum = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.where(row_sum != 0, 1.0 / torch.sqrt(row_sum), torch.zeros(1).cuda())
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt).cuda()
        adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return adj

    def row_normalize(self,mx):
        rowsum = mx.sum(dim=1)
        r_inv = torch.where(rowsum != 0, 1.0 / rowsum, torch.zeros(1).cuda())
        r_mat_inv = torch.diagflat(r_inv).cuda()
        mx = r_mat_inv @ mx
        return mx

    def preprocess_citation(self,adj, features, normalization="FirstOrderGCN"):
        features = self.row_normalize(features) 
        if normalization=="AugNormAdj":
            adj = self.aug_normalized_adjacency(adj)
        elif normalization=="NormAdj":
            adj = self.normalized_adjacency(adj)
        else:
            print("Invalid normalization technique.")
        return adj, features

    def get_affinity_matrix(self, input_data, dim, mask_list):
        data = input_data.clone().detach()
        n_samples = data.shape[0]
        
        # 1. 一次性计算所有样本对之间的距离矩阵 (N x N)
        dist_matrix = torch.cdist(data, data, p=2)

        # 2. 找到每个样本的 k 个最近邻
        # 我们加上一个很大的值到对角线，避免一个点成为自己的最近邻
        dist_matrix.fill_diagonal_(float('inf'))
        
        # 使用 topk 找到 k 个最小的距离及其索引
        k_val = min(self.k, n_samples -1)
        if k_val <= 0: # 处理只有一个样本的极端情况
            return torch.eye(n_samples, device=data.device), input_data

        nearest_distances, nearest_indices = torch.topk(dist_matrix, k_val, dim=1, largest=False)

        # 3. 构建邻接矩阵
        # 创建一个稀疏表示，然后转换为稠密矩阵
        adj = torch.zeros((n_samples, n_samples), device=data.device)
        row_indices = torch.arange(n_samples, device=data.device).view(-1, 1).expand_as(nearest_indices)
        
        # 计算相似度
        similarities = 1.0 / (1.0 + nearest_distances)
        
        # 使用索引和相似度填充邻接矩阵
        adj[row_indices, nearest_indices] = similarities.to(adj.dtype)
        
        # 确保邻接矩阵是对称的
        adj = (adj + adj.T) / 2.0
        
        # 4. 应用 mask
        # 将 mask 应用于邻接矩阵，去除无效样本的连接
        valid_mask = mask_list.view(-1, 1) * mask_list.view(1, -1)
        adj = adj * valid_mask
        
        # 5. 设置对角线为1
        adj.fill_diagonal_(1.0)

        return adj, input_data
        
    
    def SelfFilter(self,features, adj, degree, alpha):
        adj, features = self.preprocess_citation(adj, features, self.normalization)
        # features = torch.tensor(features, dtype=torch.float32) 
        features = features.detach().clone().to(torch.float32)
        emb = alpha * features
        for i in range(degree):
            features = torch.spmm(adj.to(features.dtype), features)
            emb = emb + (1-alpha)*features/degree
        return emb

    def delete_umask(self,a,umask):
        another_a = a[0][:int(sum(umask[0])),:]
        for batch_i in range(1,umask.shape[0]):
            seqlen_i = int(sum(umask[batch_i]))
            another_a = torch.cat([another_a,a[batch_i][:seqlen_i,:]],dim=0)
        return another_a

    def add_umask(self,a,umask):
        a_dim = a.shape[1]
        seqlen = umask.shape[1] #[batch, seqlen]
        seqlen_sum = int(sum(umask[0]))
        another_a = torch.cat([a[:seqlen_sum,:],torch.zeros(seqlen-seqlen_sum,a_dim).cuda()])
        another_a = torch.unsqueeze(another_a, dim=0)
        
        for batch_i in range(1,umask.shape[0]):
            seqlen_i = int(sum(umask[batch_i]))
            another_a_i = torch.cat([a[seqlen_sum:seqlen_sum+seqlen_i,:],torch.zeros(seqlen-seqlen_i,a_dim).cuda()])
            another_a_i = torch.unsqueeze(another_a_i, dim=0)
            another_a = torch.cat([another_a,another_a_i],dim=0)
            seqlen_sum = seqlen_sum+seqlen_i
            
        return another_a
    
    def MixedFilter(self, X, S, S_bar1,S_bar2, k1, k2, mu):
        H_low1 = self.LowPassFilter(X, S_bar1, k2)
        H_low2 = self.LowPassFilter(X, S_bar2, k1)
        H =  (1 - mu) * H_low1 + mu * H_low2
        return H
    
    def LowPassFilter(self, X, S, k1, p=0.5):
        I = torch.eye(S.shape[0]).cuda() 
        S = S + I
        S = self.normalize_matrix(S)
        L_S = I - S 

        H_low = X.clone() 
        for i in range(k1):
            H_low = (I - p * L_S).matmul(H_low)
            
        return H_low

    def normalize_matrix(self, A, eps=1e-12):
        D = torch.sum(A, dim=1) + eps
        D = torch.pow(D, -0.5)
        D[D == float('inf')] = 0
        D[D != D] = 0
        D = torch.diagflat(D).cuda()
        A = D @ A @ D
        return A

    def knn_fill(self,t,v,t_adj,v_adj,features_mask_del_umask):
        new_features_list = [t,v]#[2,len,dim]
        new_adj_list = [t_adj,v_adj]#[2,len,len]
        
        features_list = [t,v]#[2,len,dim]
        adj_list = [t_adj,v_adj]#[2,len,len]
        n = v.shape[0]
        
        features_len = features_mask_del_umask.shape[0] #[len,2]
        for i in range(features_len): 
            for j in range(2): 
                if features_mask_del_umask[i][j] == 0:
                    other_modality_idx = (j + 1) % 2
                    # 检查另一个模态是否存在
                    if features_mask_del_umask[i][other_modality_idx] != 0:
                        num_link = 0
                        # neighbor_indices = torch.nonzero(adj_list[other_modality_idx][i]).squeeze()
                        neighbor_indices = torch.nonzero(adj_list[other_modality_idx][i]).flatten()
                        
                        for neighbor_idx in neighbor_indices:
                            # 如果其他特征有当前缺失的模态
                            if features_mask_del_umask[neighbor_idx][j] != 0:
                                new_features_list[j][i] = new_features_list[j][i] + features_list[j][neighbor_idx]
                                num_link += 1
                        
                        if num_link != 0:
                            new_features_list[j][i] = new_features_list[j][i] / num_link

        return new_features_list[0], new_features_list[1], new_adj_list[0], new_adj_list[1]
    
    def forward(self, inputfeats, umask, input_features_mask):
        inputfeats_tensor = inputfeats[0]
        t = inputfeats_tensor[:,:,0:self.tdim].permute(1,0,2)
        v = inputfeats_tensor[:,:,self.tdim:].permute(1,0,2)
        raw_shape = t.shape
        #a:torch.Size([batch, seqlen, 512])，mask:torch.Size([batch, seqlen])
        
        features_mask = input_features_mask[0].permute(1,0,2)#torch.Size([batch, seqlen, 3]),a,t,v
        features_mask_del_umask = self.delete_umask(features_mask,umask)#torch.Size([umask_no_0, 3]),a,t,v
        # print("Shape of features_mask_del_umask:", features_mask_del_umask.shape)
        t = self.delete_umask(t,umask)
        t_adj,t = self.get_affinity_matrix(t,self.tdim,features_mask_del_umask[:,0])
        v = self.delete_umask(v,umask)
        v_adj,v = self.get_affinity_matrix(v,self.vdim,features_mask_del_umask[:,1])
        
        t,v,t_adj,v_adj = self.knn_fill(t,v,t_adj,v_adj,features_mask_del_umask)
        
        k1 = self.k1
        k2 = self.k2
        mu = self.mu
        
        F_t = self.MixedFilter(t, t_adj, v_adj, v_adj,  k1, k2, mu)
        encoded_t = self.SelfFilter(F_t, t_adj, self.degree, self.alpha)
        encoded_t = self.Wt(encoded_t)
        featureInfo_t = torch.sigmoid(self.weight_t(encoded_t))
        encoded_t = encoded_t * featureInfo_t
        
        F_v = self.MixedFilter(v, v_adj, t_adj, t_adj, k1, k2, mu)
        encoded_v = self.SelfFilter(F_v, v_adj, self.degree, self.alpha)
        encoded_v = self.Wv(encoded_v)
        featureInfo_v = torch.sigmoid(self.weight_v(encoded_v))
        encoded_v = encoded_v * featureInfo_v
        
        featureInfo_loss = torch.mean(featureInfo_t) + torch.mean(featureInfo_v)
        
        #encode:[batch, seqlen, dim]
        decoded_t = self.decoder_text(encoded_t)
        decoded_v = self.decoder_video(encoded_v)

        decoded_t = self.add_umask(decoded_t,umask)
        decoded_t = decoded_t.view(raw_shape[0],raw_shape[1],-1)
        decoded_v = self.add_umask(decoded_v,umask)
        decoded_v = decoded_v.view(raw_shape[0],raw_shape[1],-1)

        encoded_t = self.add_umask(encoded_t,umask)
        encoded_t = encoded_t.view(raw_shape[0],raw_shape[1],-1)
        encoded_v = self.add_umask(encoded_v,umask)
        encoded_v = encoded_v.view(raw_shape[0],raw_shape[1],-1)
        
        hidden = torch.cat([encoded_t,encoded_v], dim=2).permute(1,0,2)#[batch, seqlen, dim]
        reconfiguration_result = [torch.cat([decoded_t,decoded_v], dim=2).permute(1,0,2)]
        
        return reconfiguration_result,hidden,[encoded_t,encoded_v,featureInfo_loss]


class IndexFlatL2:
    def __init__(self, vectors):
        self.vectors = torch.tensor(vectors)

    def search(self, query_vector, k):
        distances = torch.norm(self.vectors - query_vector, p=2, dim=1)
        if distances.shape[0] < k:
            k = distances.shape[0]
        nearest_distances, nearest_indices = torch.topk(distances, k, largest=False)

        return nearest_indices, nearest_distances
