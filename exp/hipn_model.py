from exp.module.vocab.vocabulary import Vocabulary
from exp.module.structural.comgcn_model_dgl import CompGCN_DistMult_DGL, CompGCN_ConvE_DGL
from exp.module.structural.compgcn_model import CompGCN_TransE, CompGCN_DistMult, CompGCN_ConvE
from exp.module.temporal.temporal_attention_layer import TemporalSelfAttentionLayer
from exp.util.utils import *

class HIPN(nn.Module):
    def __init__(self, h_dim, num_nodes, num_rels, edge_index, edge_type, params, dropout=0, name='compgcn_distmult',
                 seq_len=10, num_k=100):
        super(HIPN, self).__init__()
        self.h_dim = h_dim
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_k = num_k
        self.dropout = nn.Dropout(dropout)
        self.entity_embeds = None
        self.rel_embeds = None
        self.p = params
        # ablation settings
        self.use_vocab = True
        self.use_multi_step = False
        # dgl implementation
        self.use_dgl = True
        # case study
        self.print_analysis_process = False
        # other analysis
        self.k_factor = 4
        self.seq_len = seq_len  # num windows
        # structure aggregator
        if self.use_dgl is False:
            if name == 'compgcn_transe':
                self.aggregator = CompGCN_TransE(edge_index, edge_type, params=params)
            elif name == 'compgcn_distmult':
                self.aggregator = CompGCN_DistMult(edge_index, edge_type, params=params)
            elif name == 'compgcn_conve':
                self.aggregator = CompGCN_ConvE(edge_index, edge_type, params=params)
        else:
            if name == 'compgcn_distmult':
                self.aggregator = CompGCN_DistMult_DGL(num_ent=self.num_nodes, num_rel=self.num_rels, num_base=self.p.num_bases,
                                     init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                     n_layer=self.p.gcn_layer,
                                     bias=self.p.bias, gcn_drop=self.p.dropout, opn=self.p.opn, k_factor=self.k_factor,
                                     hid_drop=self.p.hid_drop)
            elif name == 'compgcn_conve':
                self.aggregator = CompGCN_ConvE_DGL(num_ent=self.num_nodes, num_rel=self.num_rels, num_base=self.p.num_bases,
                                  init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                  n_layer=self.p.n_layer,
                                  bias=self.p.bias, gcn_drop=self.p.dropout, opn=self.p.opn,
                                  hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                  conve_hid_drop=self.p.hid_drop2, feat_drop=self.p.feat_drop,
                                  num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w, k_factor=self.k_factor)

        # temporal
        self.temporal_att = TemporalSelfAttentionLayer(h_dim, h_dim, num_nodes, seq_len)
        self.rnn = nn.GRU(h_dim, h_dim)

        #  history forward
        self.predict_r = nn.Linear(3 * h_dim, num_rels)
        self.predict_o = nn.Linear(2*h_dim, num_nodes)
        self.vocabulary = Vocabulary(num_nodes, h_dim, num_rels)

        # sigmoid
        self.sigmoid = nn.Sigmoid()
        #  loss function
        self.loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        #  evaluate_softmax
        self.evaluate_softmax = nn.Softmax(dim=0)

    def forward(self, triples, negative_triples, labels, current_time, time_unit, cache, t_evolve_r, t_len_r,
                s_d_dict_er2e, o_d_dict_er2e, graph):
        s = triples[:, 0]
        r = triples[:, 1]
        o = triples[:, 2]
        ns = negative_triples[:, 0]
        nr = negative_triples[:, 1]
        no = negative_triples[:, 2]
        alls = torch.cat((s, ns), dim=0)
        allr = torch.cat((r, nr), dim=0)
        allo = torch.cat((o, no), dim=0)
        d_dict_er2e = s_d_dict_er2e
        s_len = s.size()[0]
        distribute_er2e = torch.zeros(s_len, self.num_nodes).cuda()
        r_evolve = torch.zeros(s_len, self.seq_len, self.h_dim).cuda()
        len_r = []

        if self.use_dgl:
            embedding_loss = self.bce_loss(self.aggregator(graph, alls, allr, allo), labels)
        else:
            embedding_loss = self.loss(self.aggregator(s, r), o)
        # get the updated embeddings in SIP module
        self.entity_embeds, self.rel_embeds = self.aggregator.get_embeddings()

        if self.use_vocab:
            for i in range(s_len):
                distribute_t_er2e = torch.zeros(self.seq_len, self.num_nodes).cuda()
                row, column = d_dict_er2e[(s[i].item(), r[i].item())]
                distribute_t_er2e[row, column] = 1
                distribute_er2e[i] = torch.sum(distribute_t_er2e, dim=0)
            vocab_loss = self.vocabulary(s, r, o, distribute_er2e)
            embedding_loss += vocab_loss

        if self.use_multi_step:
            cache = torch.cat((cache,  self.entity_embeds.unsqueeze(0)))
            t_att_embeddings = self.temporal_att(cache)
            t_curr_embeddings = t_att_embeddings[self.seq_len - 1]
            cnt_non_zero = 0
            non_zero_id = []
            for i in range(s_len):
                c_r_len = t_len_r[(s[i].item(), o[i].item())]
                if c_r_len != 0:
                    non_zero_id.append(i)
                    len_r.append(c_r_len)
                    indices = t_evolve_r[(s[i].item(), o[i].item())]
                    r_evolve[cnt_non_zero][:c_r_len] = self.rel_embeds[indices]
                    cnt_non_zero = cnt_non_zero + 1
            r_evolve = r_evolve[0:cnt_non_zero]
            if cnt_non_zero > 0:
                pack = nn.utils.rnn.pack_padded_sequence(r_evolve, len_r, batch_first=True, enforce_sorted=False)
                out, h_n = self.rnn(pack)
                h_n = torch.squeeze(h_n, 0)
                lh_n = torch.zeros(s_len, self.h_dim).cuda()
                lh_n[non_zero_id] = h_n
            else:
                lh_n = torch.zeros(s_len, self.h_dim).cuda()
            r_prediction = self.predict_r(
                torch.cat((t_curr_embeddings[s], lh_n, t_curr_embeddings[o]), 1))
            r_loss = self.loss(r_prediction, r)
            embedding_loss += r_loss

        return embedding_loss

    def evaluate_filter(self, triple, current_time, time_unit, total_data, s_d_dict_er2e, o_d_dict_er2e, edge_index,
                        edge_type, cache, entity_pair, t_evolve_r, t_len_r, graph):
        s = triple[0]
        r = triple[1]
        o = triple[2]
        t = triple[3]

        if self.use_multi_step:
            step = t.item() - current_time.item()
            d_dict_er2e = self.predict_structure(s, r, step//time_unit, cache, graph, s_d_dict_er2e, entity_pair,
                                                 t_evolve_r, t_len_r)
            s_d_dict_er2e = d_dict_er2e

        if self.use_dgl:
            sub_prediction = self.aggregator.forward_raw(graph, s, r).squeeze()
        else:
            sub_prediction = self.aggregator(s, r).squeeze()
        # get the updated embeddings in SIP module
        self.entity_embeds, self.rel_embeds = self.aggregator.get_embeddings()

        s_row, s_column = s_d_dict_er2e[(s.item(), r.item())]
        s_distribute_t_er2e = torch.zeros(self.seq_len, self.num_nodes).cuda()
        s_distribute_t_er2e[s_row, s_column] = 1
        s_distribute_er2e = torch.sum(s_distribute_t_er2e, dim=0)

        if self.use_vocab:
            sub_prediction = sub_prediction + self.vocabulary.prediction(s, r, s_distribute_er2e)

        # compute_rank, filter the ground truth entities in all times as RE-NET and CyGNet
        o_label = o
        ground = sub_prediction[o].clone()
        train_filter_list = torch.nonzero(s_distribute_er2e)
        sub_prediction[train_filter_list] = 0
        s_id = torch.nonzero(total_data[:, 0] == s).view(-1)
        idx = torch.nonzero(total_data[s_id, 1] == r).view(-1)
        idx = s_id[idx]
        idx = total_data[idx, 2]
        sub_prediction[idx] = 0
        sub_prediction[o_label] = ground
        ob_pred_comp = (sub_prediction > ground).data.cpu().numpy()
        rank_ob = np.sum(ob_pred_comp) + 1
        return np.array([rank_ob])


    def predict_structure(self, subject, relation, num_steps, cache, graph, d_dict_er2e, entity_pair, t_evolve_r,
                          t_len_r):
        step = 0
        subject = subject.item()
        relation = relation.item()
        while step < 1:
            # get the updated embeddings in SIP module
            if self.use_dgl:
                self.aggregator.forward_raw(graph, torch.tensor(subject).cuda(), torch.tensor(relation).cuda()).squeeze()
            else:
                self.aggregator(torch.tensor(subject).cuda(), torch.tensor(relation).cuda()).squeeze()
            self.entity_embeds, self.rel_embeds = self.aggregator.get_embeddings()

            candidate_o = entity_pair[(subject, relation)]
            len_o = len(candidate_o)
            s_list = []
            r_list = []
            o_list = []
            candidate_triple = []
            '''
            generate candidate triples
            '''
            cache = torch.cat((cache, self.entity_embeds.unsqueeze(0)))
            if cache.size()[0] > self.seq_len:
                cache = cache[1:self.seq_len]
            t_att_embeddings = self.temporal_att(cache)
            t_curr_embeddings = t_att_embeddings[self.seq_len - 1]
            for i in range(len_o):
                len_r = []
                if (subject, candidate_o[i]) in t_len_r:
                    c_r_len = t_len_r[(subject, candidate_o[i])]
                    if c_r_len == 0:
                        h_n = torch.zeros(self.h_dim).cuda()
                    else:
                        r_evolve = torch.zeros(1, self.seq_len, self.h_dim).cuda()
                        len_r.append(c_r_len)
                        indices = torch.LongTensor(t_evolve_r[(subject, candidate_o[i])]).cuda()
                        r_evolve[0][:c_r_len] = self.rel_embeds[indices]
                        pack = nn.utils.rnn.pack_padded_sequence(r_evolve, len_r, batch_first=True, enforce_sorted=False)
                        out, h_n = self.rnn(pack)
                else:
                    h_n = torch.zeros(self.h_dim).cuda()
                r_prediction = self.predict_r(torch.cat((t_curr_embeddings[subject].squeeze(), h_n.squeeze(), t_curr_embeddings[candidate_o[i]].squeeze())))
                r_prediction.reshape(1, -1)
                r_score, r_dim = torch.max(r_prediction, 0)
                s_list.append(subject)
                r_list.append(r_dim)
                o_list.append(candidate_o[i])
                candidate_triple.append((subject, r_dim.item(), candidate_o[i]))
            '''
            generate prediction result
            '''
            s_indices = torch.LongTensor(s_list)
            r_indices = torch.LongTensor(r_list)
            o_indices = torch.LongTensor(o_list)
            res = self.sigmoid(torch.sum(self.entity_embeds[s_indices] * self.rel_embeds[r_indices] *
                                         self.entity_embeds[o_indices], dim=1))
            if self.use_vocab:
                for i in range(len_o):
                    # invalid index of a 0-dim tensor
                    if len_o == 1:
                        if (subject, r_list[i].item()) in d_dict_er2e:
                            s_row, s_column = d_dict_er2e[(subject, r_list[i].item())]
                            s_distribute_t_er2e = torch.zeros(self.seq_len, self.num_nodes).cuda()
                            s_distribute_t_er2e[s_row, s_column] = 1
                            s_distribute_er2e = torch.sum(s_distribute_t_er2e, dim=0)
                            res = self.vocabulary.prediction(subject, r_list[i], s_distribute_er2e.squeeze()[o_list[i]])
                        break
                    else:
                        if (subject, r_list[i].item()) in d_dict_er2e:
                            s_row, s_column = d_dict_er2e[(subject, r_list[i].item())]
                            s_distribute_t_er2e = torch.zeros(self.seq_len, self.num_nodes).cuda()
                            s_distribute_t_er2e[s_row, s_column] = 1
                            s_distribute_er2e = torch.sum(s_distribute_t_er2e, dim=0)
                            res[i] += self.vocabulary.prediction(subject, r_list[i], s_distribute_er2e.squeeze())[o_list[i]]

            sorted, indices = torch.sort(res.squeeze(), descending=True)
            if len_o == 1:
                indices_len = 0
            else:
                indices_len = indices.size()[0]
            min = self.num_k
            if min > indices_len:
                min = indices_len
            for i in range(min):
                cs = s_list[i]
                cr = r_list[i]
                co = o_list[i]
                if cr == relation:
                    row, column = d_dict_er2e[(subject, relation)]
                    row.append(self.seq_len-1)
                    column.append(co)
                    if (cs, co) not in t_evolve_r:
                        t_evolve_r[(cs, co)] = []
                    t_evolve_r[(cs, co)].append(cr)
                    if (cs, co) not in t_len_r:
                        t_len_r[(cs, co)] = 0
                    while len(t_evolve_r[(cs, co)]) > self.seq_len:
                        t_evolve_r[(cs, co)].pop(0)
                    t_len_r[(cs, co)] = len(t_evolve_r[(cs, co)])
                if self.print_analysis_process:
                    print("inference at time: " + str(step))
                    print(candidate_triple[indices[i]])
            step = step + 1
        return d_dict_er2e

    def reset_edge_information(self, edge_index, edge_type):
        if not self.use_dgl:
            self.aggregator.reset_edge_information(edge_index, edge_type)

    '''return self.init_embed, self.init_rel'''
    def get_embeddings(self):
        return self.aggregator.get_embeddings()




