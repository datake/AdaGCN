import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

class SparseMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.save_for_backward(sparse, dense)
        return torch.mm(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        sparse, dense = ctx.saved_tensors
        grad_sparse = grad_dense = None
        if ctx.needs_input_grad[0]:
            grad_sparse = torch.mm(grad_output, dense.t())
        if ctx.needs_input_grad[1]:
            grad_dense = torch.mm(sparse.t(), grad_output)
        return grad_sparse, grad_dense


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # initialization
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):  # softmax(A * ReLU(A X W^0) * W^1), input: X, adj: A
        support = torch.mm(input, self.weight)  # dense matrix multiplication： X * W
        output = SparseMM.apply(adj, support) # modification

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'