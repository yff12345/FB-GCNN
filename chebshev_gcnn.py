import scipy
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy
import pdb

from grad_cam import GradCam
from lib import graph
from utils import model_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
:: 输入: 一个 numpy 类型的初始 Laplacian 矩阵
:: 输出: 一个 Tensor 类型的 rescale 过的 Laplacian 稀疏张量( L_rescale = L - I )
'''


def laplacian_to_sparse(laplacian):
    """
    Rescale Laplacian and store as a torch sparse tensor. Copy to not modify the shared laplacian.
    """
    laplacian = scipy.sparse.csr_matrix(laplacian)
    laplacian = graph.rescale_L(laplacian, lamda_max=2)
    laplacian = laplacian.tocoo()  # 转换为坐标格式矩阵

    indices = torch.LongTensor(np.row_stack((laplacian.row, laplacian.col)))
    data = torch.FloatTensor(laplacian.data)
    shape = torch.Size(laplacian.shape)

    sparse_lapla = torch.sparse.FloatTensor(indices, data, shape)
    return sparse_lapla


def get_laplacians(adj_list):
    laplacians_list = []
    for adj_item in adj_list:
        laplacian = graph.laplacian(adj_item, normalized=True)
        laplacian = laplacian_to_sparse(laplacian)
        laplacians_list.append(laplacian)
    return laplacians_list


def get_bbox(x, adjs, gc_output, layer_weights, indices, rate=0.1):
    # x.shape = (100, 62, 5)
    # gc_output.shape = (100, 32, 62, 5)
    # layer_weights.shape = (100, 32, 62, 5)
    gc_cam = gc_output.clone().detach() * layer_weights  # (100, 32, 62, 5)
    mask_sum = gc_cam.sum(1, keepdim=True).sum(3, keepdim=True)    # (100, 1, 62, 1)

    batch_size, channels, nodes, features = mask_sum.size()

    mask_sum = mask_sum.view(mask_sum.size(0), -1)  # (100, 62)
    mask_max = mask_sum.max(-1, keepdim=True)[0]    # (100, 1)
    mask_min = mask_sum.min(-1, keepdim=True)[0]    # (100, 1)

    mask_sum = (mask_sum - mask_min) / (mask_max - mask_min)
    mask = torch.sign(torch.sign(mask_sum - rate) + 1)  # (100, 62)
    mask = mask.view(batch_size, nodes)       # (100, 62)

    input_box = []
    adj_input_box = []
    adj_input_box_2 = []


    for k in range(mask.size(0)):
        # indices = mask[k].nonzero().squeeze(1)
        # indices = torch.nonzero(mask[k]).squeeze(1)     # cuda
        adj_input_box.append(adj_set_zero(adjs[k], indices[k].cpu().numpy()))
        adj_input_box_2.append(sp.csr_matrix(adj_set_zero(adjs[k], indices[k].cpu().numpy())))
        tmp = x.cpu().numpy()[k, :, :]
        input_box.append(set_zero(tmp, indices[k].cpu().numpy()))

    input_box = torch.stack(input_box, dim=0)
    return input_box.cuda(), get_laplacians(adj_input_box_2), adj_input_box


# 矩阵置零
def set_zero(matrix, indices):
    input_box = np.zeros_like(matrix)   # (62, 5)
    for i in range(62):
        for item in indices:
            if i == item:
                input_box[i] = np.copy(matrix[i])
    return torch.from_numpy(input_box)


# 邻接矩阵置零
def adj_set_zero(adj_matrix, indices):
    input_box = np.zeros_like(adj_matrix)   # (62, 62)
    input_box_2 = np.zeros_like(adj_matrix)
    for i in range(62):
        for item in indices:
            if i == item:
                input_box[i] = np.copy(adj_matrix[i])
    for j in range(62):
        for item in indices:
            if j == item:
                input_box_2[:, j] = np.copy(input_box[:, j])
    return torch.from_numpy(input_box_2)


class Classifier(nn.Module):
    def __init__(self, in_panel, out_panel, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_panel, out_panel, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim()==1:
            logit =logit.unsqueeze(0)
        return logit


class ChebshevGCNN(nn.Module):
    def __init__(self, in_channels, out_channels, poly_degree, pooling_size, laplacians):
        super(ChebshevGCNN, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        self.poly_degree = poly_degree
        self.pooling_size = pooling_size
        self.laplacians = laplacians

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset weight and bias
        """
        model_utils.truncated_normal_(self.weight, mean=0.0, std=0.1)
        model_utils.truncated_normal_(self.bias, mean=0.0, std=0.1)

    def chebyshev(self, x):
        batch_size, node_num, in_features = x.size()    # (100, 62, 5)
        filter_num = self.weight.size(1)                # 卷积核个数

        '''
        new world!
        '''
        x_split = []
        for i, item in enumerate(x):     # x0.shape = (62, 5)
            x0 = item
            x_list = [x0]
            if self.poly_degree > 1:
                x1 = torch.sparse.mm(self.laplacians[i].to(DEVICE), x0)  # (62, 5)
                x_list.append(x1)
            for k in range(2, self.poly_degree):
                x2 = 2 * torch.sparse.mm(self.laplacians[i].to(DEVICE), x1) - x0  # (62, 5)
                x_list.append(x2)
                x0, x1 = x1, x2
            item = torch.stack(x_list, dim=0).permute(1, 2, 0)  # (62, 5, 25)
            x_split.append(item)
        x = torch.stack(x_split, dim=0)     # (100, 62, 5, 25)
        x = torch.reshape(x, [batch_size * node_num * in_features, self.poly_degree])   # (31000, 25)

        x = torch.matmul(x, self.weight)  # (31000, 32)
        x = torch.reshape(x, [batch_size, node_num, in_features, filter_num])  # (100, 62, 5, 32)
        x = x.permute(0, 3, 1, 2)
        return x    # (100, 32, 62, 5)

    def brelu(self, x):
        """Bias and ReLU. One bias per filter."""
        return F.relu(x + self.bias)

    def forward(self, x):
        x = self.chebyshev(x)
        x = self.brelu(x)
        return x


class FineGrainedGCNN(nn.Module):
    def __init__(self, adj, classes_num, args):
        super(FineGrainedGCNN, self).__init__()
        self.batch_size = args.batch_size
        self.filter_size = args.filter_size
        self.pooling_size = args.pooling_size
        self.poly_degree = args.poly_degree
        self.adjs_1 = [adj.toarray() for j in range(self.batch_size)]
        self.classes_num = classes_num
        laplacian = graph.laplacian(adj, normalized=True)
        laplacian = laplacian_to_sparse(laplacian)
        laplacians_1 = [laplacian for i in range(self.batch_size)]

        self.feature_num = 5
        self.laplacians_gn = None

        # --- Gating Notwork
        self.gc = ChebshevGCNN(
            in_channels=self.poly_degree[0],     # 25
            out_channels=self.filter_size[0],    # 32
            poly_degree=self.poly_degree[0],     # 25
            pooling_size=self.pooling_size[0],
            laplacians=laplacians_1
        )
        self.fc = nn.Linear(
            in_features=62 * self.filter_size[0] * self.feature_num,
            out_features=classes_num
        )

        # --- Expert 1
        self.gc_expert_1 = ChebshevGCNN(
            in_channels=self.poly_degree[0],     # 25
            out_channels=self.filter_size[0],    # 32
            poly_degree=self.poly_degree[0],     # 25
            pooling_size=self.pooling_size[0],
            laplacians=laplacians_1
        )
        self.fc_expert_1 = nn.Linear(
            in_features=62 * self.filter_size[0] * self.feature_num,
            out_features=self.classes_num
        )



    def forward(self, x, y):
        # input x.shape: (100, 62, 5)
        # --- Expert 1
        gc_output_1 = self.gc_expert_1(x)  # (100, 32, 31, 5)
        batch_size, filter_num, node_num, feature_num = gc_output_1.size()
        gc_output_1_re = torch.reshape(gc_output_1, [batch_size, filter_num * node_num * feature_num])  # (100, 9920)
        logits_expert_1 = self.fc_expert_1(gc_output_1_re)      # (100, 7)

        with torch.enable_grad():
            grad_cam = GradCam(model=self, feature_extractor=self.gc_expert_1, fc=self.fc_expert_1)
            layer_weights_1, mask_1 = grad_cam(x.detach(), y)

        input_box_1, laplacians_list_2, adjs_2 = get_bbox(x, self.adjs_1, gc_output_1, layer_weights_1, mask_1, rate=0.3)

        # --- Expert 2
        self.gc_expert_2 = ChebshevGCNN(
            in_channels=self.poly_degree[0],    # 25
            out_channels=self.filter_size[0],   # 32
            poly_degree=self.poly_degree[0],    # 25
            pooling_size=self.pooling_size[0],
            laplacians=laplacians_list_2
        ).to(DEVICE)
        self.fc_expert_2 = nn.Linear(
            in_features=62 * self.filter_size[0] * self.feature_num,
            out_features=self.classes_num
        ).to(DEVICE)

        gc_output_2 = self.gc_expert_2(input_box_1)  # (100, 32, 62, 5)
        batch_size, filter_num, node_num, feature_num = gc_output_2.size()
        gc_output_2_re = torch.reshape(gc_output_2, [batch_size, filter_num * node_num * feature_num])
        logits_expert_2 = self.fc_expert_2(gc_output_2_re)  # (100, 7)

        with torch.enable_grad():
            grad_cam = GradCam(model=self, feature_extractor=self.gc_expert_2, fc=self.fc_expert_2)
            layer_weights_2, mask_2 = grad_cam(input_box_1.detach(), y)

        input_box_2, laplacians_list_3, adjs_3 = get_bbox(input_box_1, adjs_2, gc_output_2, layer_weights_2, mask_2, rate=0.3)

        # --- Expert 3
        self.gc_expert_3 = ChebshevGCNN(
            in_channels=self.poly_degree[0],  # 25
            out_channels=self.filter_size[0],  # 32
            poly_degree=self.poly_degree[0],  # 25
            pooling_size=self.pooling_size[0],
            laplacians=laplacians_list_3
        ).to(DEVICE)
        self.fc_expert_3 = nn.Linear(
            in_features=62 * self.filter_size[0] * self.feature_num,
            out_features=self.classes_num
        ).to(DEVICE)
        gc_output_3 = self.gc_expert_3(input_box_2)  # (100, 32, 31, 5)
        batch_size, filter_num, node_num, feature_num = gc_output_3.size()
        gc_output_3_re = torch.reshape(gc_output_3, [batch_size, filter_num * node_num * feature_num])  # (100, 9920)
        logits_expert_3 = self.fc_expert_3(gc_output_3_re)  # (100, 7)

        # --- Gating Network
        my_gate = self.gc(x)
        batch_size, filter_num, node_num, feature_num = my_gate.size()
        my_gate = torch.reshape(my_gate, [batch_size, filter_num * node_num * feature_num])  # (100, 9920)
        my_gate = self.fc(my_gate)
        pr_gate = F.softmax(my_gate, dim=1)  # (100, 7)

        logits_gate = torch.stack([logits_expert_1, logits_expert_2, logits_expert_3], dim=-1)  # (100, 7, 3)
        logits_gate = logits_gate * pr_gate.view(pr_gate.size(0), pr_gate.size(1), 1)
        logits_gate = logits_gate.sum(-1)

        return logits_gate