import pickle
import numpy as np
import os
import dgl
import torch
import torch.nn as nn


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)

def make_batch(a, b, c, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n], b[i:i+n], c[i:i+n]

def make_train_batch(a, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i + n]

def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.apply_edges(lambda edges: {'norm_s': edges.dst['norm'] * edges.src['norm']})
    g.apply_edges(lambda edges: {'norm_o': edges.src['norm'] * edges.dst['norm']})
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    g.to("cpu")
    return g

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def get_data(s_hist, o_hist):
    data = None
    for i, s_his in enumerate(s_hist):
        if len(s_his) != 0:
            tem = torch.cat((torch.LongTensor([i]).repeat(len(s_his), 1), torch.LongTensor(s_his.cpu())), dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)

    for i, o_his in enumerate(o_hist):
        if len(o_his) != 0:
            tem = torch.cat((torch.LongTensor(o_his[:,1].cpu()).view(-1,1), torch.LongTensor(o_his[:,0].cpu()).view(-1,1), torch.LongTensor([i]).repeat(len(o_his), 1)), dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)
    data = np.unique(data, axis=0)
    return data

def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    cuda_id = torch.cuda.current_device()
    g = g.to('cuda:' + str(cuda_id))
    return g

# assuming pred and soft_targets are both Variables with shape (batchsize, num_of_classes), each row of pred is predicted logits and each row of soft_targets is a discrete distribution.
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax()
    pred = pred.type('torch.DoubleTensor').cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

# get the entity distribution for each t
def get_true_distribution(train_data, num_s):
    true_s = np.zeros(num_s)
    true_o = np.zeros(num_s)
    true_prob_s = None
    true_prob_o = None
    current_t = 0
    for triple in train_data:
        s = triple[0]
        o = triple[2]
        t = triple[3]
        true_s[s] += 1
        true_o[o] += 1
        if current_t != t:
            true_s = true_s / np.sum(true_s)
            true_o = true_o / np.sum(true_o)
            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_s)
                true_prob_o = true_o.reshape(1, num_s)
            else:
                # array concat
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_s)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_s)), axis=0)

            true_s = np.zeros(num_s)
            true_o = np.zeros(num_s)
            current_t = t
    true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_s)), axis=0)
    true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_s)), axis=0)
    return true_prob_s, true_prob_o

def generate_negative_triples(b_data, len, negative_ratio, number_entities):
    negative_obj = torch.LongTensor(len*negative_ratio).random_(number_entities - 1).squeeze()
    sub_n_data = b_data.repeat(negative_ratio, 1)
    sub_n_data[:, 2] = negative_obj

    negative_obj_reverse = torch.LongTensor(len * negative_ratio).random_(number_entities - 1).squeeze()
    ob_n_data = b_data.repeat(negative_ratio, 1)
    ob_n_data[:, 2] = negative_obj_reverse
    return torch.cat((sub_n_data, ob_n_data), dim=0)

def get_graph_dict(triple_dict, train_data, train_times, test_data, test_times, dev_data=None, dev_times=None):
    for tim in train_times:
        data = get_data_with_t(train_data, tim)
        triple_dict[tim] = data
    if dev_data is not None:
        for tim in dev_times:
            data = get_data_with_t(dev_data, tim)
            triple_dict[tim] = data
    for tim in test_times:
        data = get_data_with_t(test_data, tim)
        triple_dict[tim] = data
    return triple_dict

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

def convert_list2graph(h_dict, num_rels):
    g_dict = {}
    for key in h_dict:
        g_list = []
        for triples in h_dict[key]:
            t = np.array(triples)
            g = get_big_graph(t, num_rels)
            g_list.append(g)
        g_dict[key] = g_list
    return g_dict

def get_history_all_distribution_before_train(triple_dict, num_k, stop_time, dataset):

    if os.path.exists('./data/' + dataset + '/all_his_d'):
        with open('./data/' + dataset + '/all_his_d', 'rb') as f:
            return pickle.load(f)

    all_his_dict_sub_er2e = {}
    all_his_dict_ob_er2e = {}
    current_dict_sub_sr_er2e = {}
    current_dict_ob_sr_er2e = {}
    h_dict_sub_sr_er2e = {}
    h_dict_ob_sr_er2e = {}

    for key in triple_dict:
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        er2e_c_dict = {}
        er2e_has_done = set()
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            # entity relation to entity
            if (s, r) not in h_dict_sub_sr_er2e:
                h_dict_sub_sr_er2e[(s, r)] = []
            if (s, r) not in current_dict_sub_sr_er2e:
                current_dict_sub_sr_er2e[(s, r)] = []
            if (s, r) not in er2e_c_dict:
                er2e_c_dict[(s, r)] = []

            # entity relation to entity
            if (o, r) not in h_dict_ob_sr_er2e:
                h_dict_ob_sr_er2e[(o, r)] = []
            if (o, r) not in current_dict_ob_sr_er2e:
                current_dict_ob_sr_er2e[(o, r)] = []
            if (o, r) not in er2e_c_dict:
                er2e_c_dict[(o, r)] = []
            if key <= stop_time:
                er2e_c_dict[(s, r)].append(o)
                er2e_c_dict[(o, r)].append(s)
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            # entity relation to entity
            if (s, r) not in er2e_has_done:
                current_dict_sub_sr_er2e[(s, r)].append(er2e_c_dict[(s, r)])
                er2e_has_done.add((s, r))
                while len(current_dict_sub_sr_er2e[(s, r)]) > num_k:
                    tset = set(current_dict_sub_sr_er2e[(s, r)].pop(0))
                    tset.update(current_dict_sub_sr_er2e[(s, r)][0])
                    current_dict_sub_sr_er2e[(s, r)][0] = list(tset)
            if (o, r) not in er2e_has_done:
                current_dict_ob_sr_er2e[(o, r)].append(er2e_c_dict[(o, r)])
                er2e_has_done.add((o, r))
                while len(current_dict_ob_sr_er2e[(o, r)]) > num_k:
                    tset = set(current_dict_ob_sr_er2e[(o, r)].pop(0))
                    tset.update(current_dict_ob_sr_er2e[(o, r)][0])
                    current_dict_ob_sr_er2e[(o, r)][0] = list(tset)
        er2e_tempt_s = {}
        er2e_tempt_o = {}
        # entity relation to entity
        for s_key in h_dict_sub_sr_er2e:
            s_len = len(h_dict_sub_sr_er2e[s_key])
            index = num_k - s_len
            l1 = []
            l2 = []
            for each_time_list in h_dict_sub_sr_er2e[s_key]:
                for each_o in each_time_list:
                    l1.append(index)
                    l2.append(each_o)
                index = index + 1
            er2e_tempt_s[s_key] = (l1, l2)
        for o_key in h_dict_ob_sr_er2e:
            o_len = len(h_dict_ob_sr_er2e[o_key])
            index = num_k - o_len
            l1 = []
            l2 = []
            for each_time_list in h_dict_ob_sr_er2e[o_key]:
                for each_s in each_time_list:
                    l1.append(index)
                    l2.append(each_s)
                index = index + 1
            er2e_tempt_o[o_key] = (l1, l2)
        # entity relation to entity
        all_his_dict_sub_er2e[key] = er2e_tempt_s
        all_his_dict_ob_er2e[key] = er2e_tempt_o
        h_dict_sub_sr_er2e = current_dict_sub_sr_er2e
        h_dict_ob_sr_er2e = current_dict_ob_sr_er2e

    with open('./data/' + dataset + '/all_his_d', 'wb') as f:
        pickle.dump([all_his_dict_sub_er2e, all_his_dict_ob_er2e], f)

    return all_his_dict_sub_er2e, all_his_dict_ob_er2e

def construct_graph_for_gcn_for_each_entity(triple_dict, num_rels, num_k):
    all_his_graph = {}
    current_graph = {}
    for key in triple_dict:
        if key not in all_his_graph:
            all_his_graph[key] = []
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        all_ent = set()
        tempt_graph = {}
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            all_ent.add(s)
            all_ent.add(o)
            if s not in all_his_graph:
                all_his_graph[key][s] = []
            if o not in all_his_graph:
                all_his_graph[key][o] = []
            if s not in current_graph:
                current_graph[s] = []
            if o not in current_graph:
                current_graph[o] = []
            if s not in tempt_graph:
                tempt_graph[s] = []
            if o not in tempt_graph:
                tempt_graph[o] = []
            tempt_graph[s].append((s, r, o))
            tempt_graph[o].append((s, r, o))

        for ent in all_ent:
            all_his_graph[key][ent] = current_graph[ent]

        for ent in all_ent:
            current_graph[ent].append(construct_adj(tempt_graph[ent], num_rels))
            while len(current_graph[ent]) > num_k:
                current_graph[ent].pop(0)
    return all_his_graph

def construct_graph_for_gcn_for_each_time(triple_dict, num_rels, stop_time, dataset):
    all_edge_index = {}
    all_edge_type = {}
    if os.path.exists('./data/' + dataset + '/adj'):
        with open('./data/' + dataset + '/adj', 'rb') as f:
            return pickle.load(f)
    for key in triple_dict:
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        data = []
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            data.append((s, r, o))
        edge_index, edge_type = construct_adj(data, num_rels)
        if key == 0:
            all_edge_index[key] = edge_index
            all_edge_type[key] = edge_type
        else:
            all_edge_index[key] = tempt_edge_index
            all_edge_type[key] = tempt_edge_type
        if key < stop_time:
            tempt_edge_index = edge_index
            tempt_edge_type = edge_type
    with open('./data/' + dataset + '/adj', 'wb') as f:
        return pickle.dump([all_edge_index, all_edge_type], f)
    return all_edge_index, all_edge_type

def construct_adj(data, num_rels):
    """
    Constructor of the runner class

    Parameters
    ----------

    Returns
    -------
    Constructs the adjacency matrix for GCN

    """
    edge_index, edge_type = [], []

    for sub, rel, obj in data:
        edge_index.append((sub, obj))
        edge_type.append(rel)

    # Adding inverse edges
    for sub, rel, obj in data:
        edge_index.append((obj, sub))
        edge_type.append(rel + num_rels)

    return edge_index, edge_type

def get_active_object_dict(triple_dict, stop_time, dataset):
    active_dict = {}
    active = set()
    rel2ent = {}
    ent2ent = {}
    if os.path.exists('./data/' + dataset + '/act_obj'):
        with open('./data/' + dataset + '/act_obj', 'rb') as f:
            return pickle.load(f)
    for key in triple_dict:
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            if key == stop_time:
                active.add(s)
                active.add(o)
            if r not in rel2ent:
                rel2ent[r] = set()
            if s not in ent2ent:
                ent2ent[s] = set()
            if o not in ent2ent:
                ent2ent[o] = set()
            if key < stop_time:
                rel2ent[r].add(s)
                rel2ent[r].add(o)
                ent2ent[s].add(o)
                ent2ent[o].add(s)
            if key > stop_time:
                if (s, r) not in active_dict:
                    active_dict[(s, r)] = list(ent2ent[s].intersection(rel2ent[r]).intersection(active))
    with open('./data/' + dataset + '/act_obj', 'wb') as f:
        pickle.dump(active_dict, f)
    return active_dict

def get_entity_pair_dict(triple_dict, stop_time, dataset):
    entity_pair_dict = {}
    rel2ent = {}
    ent2ent = {}
    if os.path.exists('./data/' + dataset + '/ent_pair'):
        with open('./data/' + dataset + '/ent_pair', 'rb') as f:
            return pickle.load(f)
    for key in triple_dict:
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            if r not in rel2ent:
                rel2ent[r] = set()
            if s not in ent2ent:
                ent2ent[s] = set()
            if o not in ent2ent:
                ent2ent[o] = set()
            if key < stop_time:
                rel2ent[r].add(s)
                rel2ent[r].add(o)
                ent2ent[s].add(o)
                ent2ent[o].add(s)
            if key >= stop_time:
                if (s, r) not in entity_pair_dict:
                    entity_pair_dict[(s, r)] = list(ent2ent[s].intersection(rel2ent[r]))
    with open('./data/' + dataset + '/ent_pair', 'wb') as f:
        pickle.dump(entity_pair_dict, f)
    return entity_pair_dict

def get_evolve_r(triple_dict, num_k, num_rels, stop_time, dataset):
    evolve_r = dict()
    len_r = dict()
    tempt = dict()
    if os.path.exists('./data/' + dataset + '/evolve'):
        with open('./data/' + dataset + '/evolve', 'rb') as f:
            return pickle.load(f)
    for key in triple_dict:
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        evolve_r[key] = dict()
        len_r[key] = dict()
        done = set()
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            if (s, o) not in evolve_r[key]:
                evolve_r[key][(s, o)] = []

            if (s, o) not in len_r[key]:
                len_r[key][(s, o)] = 0

            if (s, o) not in tempt:
                tempt[(s, o)] = []

        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            if (s, o) not in done:
                done.add((s, o))
                while len(tempt[(s, o)]) > num_k:
                    tempt[(s, o)].pop(0)
                len_r[key][(s, o)] = len(tempt[(s, o)])
                evolve_r[key][(s, o)] = tempt[(s, o)][:]
            if key < stop_time:
                tempt[(s, o)].append(r)

    with open('./data/' + dataset + '/evolve', 'wb') as f:
        pickle.dump([evolve_r, len_r], f)

    return evolve_r, len_r

def get_edge_for_dgl(triple_dict, num_rel, stop_time, dataset):
    src_dict = {}
    dst_dict = {}
    rel_type = {}
    rel_type_inverse = {}
    rel_num_dict = {}
    if os.path.exists('./data/' + dataset + '/rel'):
        with open('./data/' + dataset + '/rel', 'rb') as f:
            return pickle.load(f)
    for key in triple_dict:
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        src = []
        dst = []
        r_type = []
        r_type_inverse = []
        edge_num = 0
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            src.append(s)
            dst.append(o)
            r_type.append(r)
            r_type_inverse.append(r+num_rel)
            edge_num = edge_num + 1

        src_dict[key] = src
        dst_dict[key] = dst
        rel_type[key] = r_type
        rel_type_inverse[key] = r_type_inverse
        rel_num_dict[key] = edge_num
        if key == stop_time:
            break

    with open('./data/' + dataset + '/rel', 'wb') as f:
        pickle.dump([src_dict, dst_dict, rel_type, rel_type_inverse, rel_num_dict], f)

    return src_dict, dst_dict, rel_type, rel_type_inverse, rel_num_dict

def get_first_graph(src, dst, rel, rel_inverse, num_nodes):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)  # src -> tgt
    g.add_edges(dst, src)  # tgt -> src
    norm = comp_deg_norm(g)
    g.ndata.update({'norm': norm.view(-1, 1)})
    g.edata['type'] = torch.LongTensor(rel + rel_inverse)
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.to("cpu")
    return g

def update_graph(g: dgl.DGLGraph, src, dst, rel, rel_inverse, num_edges):
    rem = []
    for i in range(2*num_edges):
        rem.append(i)
    g.remove_edges(rem)
    g.add_edges(src, dst)  # src -> tgt
    g.add_edges(dst, src)  # tgt -> src
    norm = comp_deg_norm(g)
    g.ndata.update({'norm': norm.view(-1, 1)})
    g.edata['type'] = torch.LongTensor(rel + rel_inverse).cuda()
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    return g

def get_vocabulary(triple_dict, stop_time, dataset):

    if os.path.exists('./data/' + dataset + '/all_his_d2'):
        with open('./data/' + dataset + '/all_his_d2', 'rb') as f:
            return pickle.load(f)

    all_his_dict_sub_er2e = {}
    all_his_dict_ob_er2e = {}
    current_dict_sub_sr_er2e = {}
    current_dict_ob_sr_er2e = {}
    h_dict_sub_sr_er2e = {}
    h_dict_ob_sr_er2e = {}

    for key in triple_dict:
        triples = triple_dict[key]
        triples = torch.from_numpy(triples)
        triples_length = triples.size()[0]
        er2e_c_dict = {}
        er2e_has_done = set()
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            # entity relation to entity
            if (s, r) not in er2e_c_dict:
                er2e_c_dict[(s, r)] = set()
            if (s, r) not in current_dict_sub_sr_er2e:
                current_dict_sub_sr_er2e[(s, r)] = []
            if (s, r) not in er2e_c_dict:
                er2e_c_dict[(s, r)] = []

            # entity relation to entity
            if (o, r) not in er2e_c_dict:
                er2e_c_dict[(o, r)] = set()
            if (o, r) not in current_dict_ob_sr_er2e:
                current_dict_ob_sr_er2e[(o, r)] = []
            if (o, r) not in er2e_c_dict:
                er2e_c_dict[(o, r)] = []
            if key < stop_time:
                er2e_c_dict[(s, r)].add(o)
                er2e_c_dict[(o, r)].add(s)
        for i in range(triples_length):
            s = triples[i][0].item()
            r = triples[i][1].item()
            o = triples[i][2].item()
            # entity relation to entity
            if (s, r) not in er2e_has_done:
                er2e_has_done.add((s, r))
                current_dict_sub_sr_er2e[(s, r)] = list(er2e_c_dict[(s, r)])
            if (o, r) not in er2e_has_done:
                er2e_has_done.add((o, r))
                current_dict_sub_sr_er2e[(o, r)] = list(er2e_c_dict[(o, r)])

        # entity relation to entity
        all_his_dict_sub_er2e[key] = h_dict_sub_sr_er2e
        all_his_dict_ob_er2e[key] = h_dict_ob_sr_er2e
        h_dict_sub_sr_er2e = current_dict_sub_sr_er2e
        h_dict_ob_sr_er2e = current_dict_ob_sr_er2e
    with open('./data/' + dataset + '/all_his_d2', 'wb') as f:
        pickle.dump([all_his_dict_sub_er2e, all_his_dict_ob_er2e], f)

    return all_his_dict_sub_er2e, all_his_dict_ob_er2e