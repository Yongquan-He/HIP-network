import torch
from torch import nn
import dgl
import dgl.function as fn

class DCompGCNCovLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k_factor, act=None, bias=True, drop_rate=0., opn='mult', num_base=-1,
                 num_rel=None):
        super(DCompGCNCovLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn

        # k related parameter
        self.k_factor = k_factor
        self.hyper_plane_node_w = self.get_param([self.k_factor, in_channels, in_channels // self.k_factor])
        self.hyper_plane_node_rel_w = self.get_param([self.k_factor, in_channels, in_channels // self.k_factor])
        self.hyper_plane_in_w = self.get_param(
            [self.k_factor, in_channels // self.k_factor, out_channels // self.k_factor])
        self.hyper_plane_out_w = self.get_param(
            [self.k_factor, in_channels // self.k_factor, out_channels // self.k_factor])

        # loop-relation-type specific parameter
        self.loop_w = self.get_param([in_channels, out_channels])
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        # attention related parameter
        self.att_w = self.get_param([2*in_channels//self.k_factor, 1])
        self.relu = torch.nn.ReLU()
        self.soft_max = nn.Softmax(dim=1)

        # transform embedding of relations to next layer
        self.w_rel = self.get_param([in_channels, out_channels])

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):

        edge_type = edges.data['type']

        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # coming activation information
        # transform activation(h, r)
        edge_data = torch.matmul(edge_data, self.hyper_plane_node_rel_w)  # k*E*in/k
        edge_data = edge_data.permute(1, 0, 2)  # E*k*in/k
        # transform node
        node_data = torch.matmul(edges.dst['h'], self.hyper_plane_node_w)  # k*E*in/k
        node_data = node_data.permute(1, 0, 2)  # E*k*in/k

        # NOTE: first half edges are all in-directions, last half edges are out-directions.

        cat_data = torch.cat([edge_data, node_data], 2)
        att_data = torch.matmul(cat_data, self.att_w)
        att_data = self.relu(att_data)  # E*k
        att_data = self.soft_max(att_data)  # E*k
        att_data = att_data.repeat(1, 1, self.in_channels // self.k_factor)  # E*k*in/k
        edge_data = edge_data * att_data
        edge_data = edge_data.permute(1, 0, 2)  # k*E*in/k
        msg = torch.cat([torch.bmm(edge_data[:, :edge_num // 2, :], self.hyper_plane_in_w),
                         torch.bmm(edge_data[:, edge_num // 2:, :], self.hyper_plane_out_w)], dim=1)
        msg = msg.permute(1, 0, 2)  # E*k*out/k
        msg = msg.reshape(edge_num, self.out_channels)  # E*out
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]

        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, node_repr, rel_repr):
        """
        :param g: dgl Graph, a graph without self-loop
        :param node_repr: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        """
        g.ndata['h'] = node_repr
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        g.ndata['h'] = g.ndata['h'] + torch.mm(self.comp(g.ndata['h'], self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            g.ndata['h'] = g.ndata['h'] + self.bias
        node_repr = self.bn(g.ndata['h'])
        node_repr = self.act(node_repr)
        return node_repr, torch.matmul(self.rel, self.w_rel)