import torch
from torch import nn

class TemporalSelfAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes, num_windows):
        super(TemporalSelfAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.num_windows = num_windows
        self.soft_max = nn.Softmax(dim=2)
        self.query_attention = self.get_param([in_dim, out_dim])
        self.key_attention = self.get_param([in_dim, out_dim])
        self.value_attention = self.get_param([in_dim, out_dim])
        self.time_mask = torch.zeros([num_windows, num_windows], requires_grad=False) - 1e10
        i = 0
        while i < num_windows:
            j = 0
            while j <= i:
                self.time_mask[i][j] = 0
                j = j + 1
            i = i + 1
        self.time_mask = self.time_mask.repeat(num_nodes, 1, 1).cuda()
    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def forward(self, all_time_embeddings):
        # all_time_embeddings : num_windows * num_nodes * in_channels
        all_time_embeddings = all_time_embeddings.permute(1, 0, 2)  # num_nodes * num_windows * in_channels
        # att_ent : num_nodes * num_windows * out_channels
        q_att_ent = torch.matmul(all_time_embeddings, self.query_attention)
        k_att_ent = torch.matmul(all_time_embeddings, self.key_attention)
        v_att_ent = torch.matmul(all_time_embeddings, self.value_attention)
        k_att_ent = k_att_ent.permute(0, 2, 1)   # num_nodes * out_channels * num_windows
        div = self.num_windows ** 0.5
        qk = torch.bmm(q_att_ent, k_att_ent) / div + self.time_mask  # num_nodes * num_windows * num_windows
        weight = self.soft_max(qk)  # num_nodes * num_windows * num_windows
        att_embeddings = torch.bmm(weight, v_att_ent)  # num_nodes * num_windows * out_channels
        att_embeddings = att_embeddings.permute(1, 0, 2)  # num_windows * num_nodes * out_channels
        return att_embeddings