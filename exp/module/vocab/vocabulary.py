import torch
from torch import nn
import torch.nn.functional as F

class Vocabulary(nn.Module):
    def __init__(self, i_dim, h_dim, num_rels):
        super(Vocabulary, self).__init__()

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.num_rels = num_rels

        self.h_ent_embeds = nn.Parameter(torch.Tensor(i_dim, h_dim))
        self.h_rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = 0.5

        self.W_mlp = nn.Linear(h_dim * 2, i_dim)
        self.reset_parameters()

        self.drop = torch.nn.Dropout(self.dropout)  # hidden layer dropout

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.h_ent_embeds,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.h_rel_embeds,
                                gain=nn.init.calculate_gain('relu'))

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.h_rel_embeds.pow(2)) + torch.mean(self.h_ent_embeds.pow(2))
        return regularization_loss * reg_param

    def forward(self, s, r, o, vocabulary):
        s_emb = self.h_ent_embeds[s]
        r_emb = self.h_rel_embeds[r]
        m_t = torch.cat((s_emb, r_emb), dim=1)
        score_c = vocabulary
        score_g = self.W_mlp(self.drop(m_t))
        score_g = F.softmax(score_g, dim=1)
        score = score_g + score_c
        score = torch.log(score)
        loss = F.nll_loss(score, o) + self.regularization_loss(reg_param=0.01)
        return loss

    def prediction(self, s, r, vocabulary):
        s_emb = self.h_ent_embeds[s]
        r_emb = self.h_rel_embeds[r]
        m_t = torch.cat((s_emb, r_emb), dim=0)
        score_g = self.W_mlp(m_t)
        score_c = vocabulary
        score_g = F.softmax(score_g, dim=0)
        score = score_g + score_c
        return score