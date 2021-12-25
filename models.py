import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution
from ppnp.pytorch.utils import MixedDropout, MixedLinear
import math

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, dropout_adj, layer=2):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid) # n_feature: C, n_hidden: H
        self.gc_layer = GraphConvolution(nhid, nhid)  # n_hidden: H, n_hidden: H
        self.gc2 = GraphConvolution(nhid, nclass) # n_hidden: H, n_classes: F
        self.dropout = dropout
        self.layer = layer
        self.dropout_adj = MixedDropout(dropout_adj)

    def forward(self, x, adj, idx): # X, A
        x = F.relu(self.gc1(x, self.dropout_adj(adj))) # for APPNP paper
        for i in range(self.layer - 2):
            x = F.relu(self.gc_layer(x, adj))  # middle conv
            x = F.dropout(x, self.dropout, training=self.training)
        if self.layer > 1:
            x = self.gc2(x, adj) # 2th conv
        return F.log_softmax(x, dim=1)[idx] # N * F

class AdaGCN(nn.Module):
    def __init__(self, nfeat,  nhid, nclass, dropout, dropout_adj):
        super().__init__()
        fcs = [MixedLinear(nfeat, nhid, bias=False)]
        fcs.append(nn.Linear(nhid, nclass, bias=False))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())

        if dropout is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout) # p: drop rate
        if dropout_adj is 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj) # p: drop rate
        self.act_fn = nn.ReLU()

    def _transform_features(self, x):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))
        return res

    def forward(self, x, adj, idx):  # X, A
        logits = self._transform_features(x) # MLP: X->H, Mixed-layer + some layers FC
        return F.log_softmax(logits, dim=-1)[idx]
