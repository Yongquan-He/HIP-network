import torch
from torch import nn
import dgl
from exp.module.structural.comp_conv_dgl import DCompGCNCovLayer
import torch.nn.functional as F

class CompGCN(nn.Module):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                 conv_bias=True, gcn_drop=0., opn='mult', k_factor=4):
        super(CompGCN, self).__init__()
        self.act = torch.tanh
        self.loss = nn.BCELoss()
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.n_layer = n_layer

        self.init_embed = self.get_param([self.num_ent, self.init_dim])  # initial embedding for entities
        if self.num_base > 0:
            # linear combination of a set of basis vectors
            self.init_rel = self.get_param([self.num_base, self.init_dim])
        else:
            # independently defining an embedding for each relation
            self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])

        self.conv1 = DCompGCNCovLayer(self.init_dim, self.gcn_dim, k_factor, self.act, conv_bias, gcn_drop, opn, num_base=self.num_base,
                                num_rel=self.num_rel)
        self.conv2 = DCompGCNCovLayer(self.init_dim, self.gcn_dim, k_factor, self.act, conv_bias, gcn_drop, opn, num_base=self.num_base,
                                num_rel=self.num_rel) if n_layer == 2 else None
        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def forward_base(self, g, subj, rel, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations
        x, r = self.conv1(g, x, r)
        x = drop1(x)  # embeddings of entities [num_ent, dim]
        x, r = self.conv2(g, x, r) if self.n_layer == 2 else (x, r)
        x = drop2(x) if self.n_layer == 2 else x
        sub_emb = torch.index_select(x, 0, subj)  # filter out embeddings of subjects in this batch
        rel_emb = torch.index_select(r, 0, rel)  # filter out embeddings of relations in this batch

        return sub_emb, rel_emb, x

    def get_embeddings(self):
        return self.init_embed, self.init_rel

class CompGCN_DistMult_DGL(CompGCN):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                 bias=True, gcn_drop=0., opn='mult', k_factor=4, hid_drop=0.):
        super(CompGCN_DistMult_DGL, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                              bias, gcn_drop, opn, k_factor)
        self.drop = nn.Dropout(hid_drop)

    def forward_raw(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb  # [batch_size, emb_dim]
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

    def forward(self, g, subj, rel, obj):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb  # [batch_size, emb_dim]
        x = torch.sum(obj_emb*all_ent[obj], dim=1)  # [batch_size, ent_num]
        score = torch.sigmoid(x)
        return score


class CompGCN_ConvE_DGL(CompGCN):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, k_factor=4):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(CompGCN_ConvE_DGL, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                            bias, gcn_drop, opn, k_factor)
        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)  # fully connected projection

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        assert self.embed_dim == self.k_h * self.k_w
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input

    def forward_raw(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(g, subj, rel, self.drop, self.input_drop)
        stack_input = self.concat(sub_emb, rel_emb)  # [batch_size, 1, 2*k_h, k_w]
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

    def forward(self, g, sub, rel, obj):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(g, sub, rel, self.drop, self.input_drop)
        stack_input = self.concat(sub_emb, rel_emb)  # [batch_size, 1, 2*k_h, k_w]
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.sum(x*all_ent[obj], dim=1)  # [batch_size, ent_num]
        score = torch.sigmoid(x)
        return score