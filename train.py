import argparse

import numpy as np
import time
import torch
import torch.nn as nn

from exp.hipn_model import HIPN
from exp.util import utils
import os
import pickle


def train(args):
    print("1th step---compute the number of entities and relations...")
    num_nodes, num_rels = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    # key:time value:triples
    triple_dict = {}
    args.num_ent = num_nodes
    args.num_rel = num_rels
    print("2nd step---load data...")
    if args.dataset == 'ICEWS14':
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
    else:
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'valid.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt', 'test.txt')
    save_model = './models/' + args.dataset + 'model.pt'
    print("3rd step---load triple(key: time, value: triple set)...")
    if args.dataset == 'ICEWS14':
        if os.path.exists('./data/' + args.dataset + '/triple.txt'):
            with open('./data/' + args.dataset + '/triple.txt', 'rb') as f:
                triple_dict = pickle.load(f)
        else:
            triple_dict = utils.get_graph_dict(triple_dict, train_data, train_times, test_data, test_times)
            with open('./data/' + args.dataset + '/triple.txt', 'wb') as fp:
                pickle.dump(triple_dict, fp)
    else:
        if os.path.exists('./data/' + args.dataset + '/triple.txt'):
            with open('./data/' + args.dataset + '/triple.txt', 'rb') as f:
                triple_dict = pickle.load(f)
        else:
            triple_dict = utils.get_graph_dict(triple_dict, train_data, train_times, test_data, test_times, valid_data,
                                               valid_times)
            with open('./data/' + args.dataset + '/triple.txt', 'wb') as fp:
                pickle.dump(triple_dict, fp)

    print("4th step---check cuda...")
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # solid seed
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    print("5th step---load the cache history data for speed ...")
    valid_data = torch.from_numpy(valid_data)
    test_data = torch.from_numpy(test_data)
    total_data = torch.from_numpy(total_data)
    all_his_dict_sub_er2e, all_his_dict_ob_er2e = utils.get_history_all_distribution_before_train(triple_dict, args.seq_len, valid_data[0][3].item(), args.dataset)
    sub_vocab, ob_vocab = utils.get_vocabulary(triple_dict, valid_data[0][3].item(), args.dataset)
    entity_pair_dict = utils.get_entity_pair_dict(triple_dict, valid_data[0][3].item(), args.dataset)
    all_edge_index, all_edge_type = utils.construct_graph_for_gcn_for_each_time(triple_dict, num_rels,valid_data[0][3].item(), args.dataset)
    evolve_r, len_r = utils.get_evolve_r(triple_dict, args.seq_len, num_rels, valid_data[0][3].item(), args.dataset)
    src_dict, dst_dict, rel_type, rel_type_inverse, rel_num_dict = utils.get_edge_for_dgl(triple_dict, num_rels, valid_data[0][3].item(), args.dataset)
    if use_cuda:
        total_data = total_data.cuda()

    print("6th step---init HIPN...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset, exist_ok=True)
    edge_index = all_edge_index[train_data[0][3]]
    edge_type = all_edge_type[train_data[0][3]]
    edge_index = torch.LongTensor(edge_index).cuda().t()
    edge_type = torch.LongTensor(edge_type).cuda()
    model = HIPN(args.n_hidden, num_nodes, num_rels, edge_index, edge_type, args,
                 dropout=args.dropout, name=args.comp_decoder,
                 seq_len=args.seq_len, num_k=args.num_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)
    if use_cuda:
        model.cuda()

    epoch = 0
    best_mrr = 0
    start_time = train_data[0][3].item()
    time_unit = train_times[1] - train_times[0]
    shape = [args.seq_len - 1, num_nodes, args.n_hidden]
    cache = torch.Tensor(*shape).cuda()
    nn.init.xavier_normal_(cache, gain=nn.init.calculate_gain('relu'))

    print('7th step---start training... using time unit {0}'.format(time_unit))
    while True:
        graph = utils.get_first_graph(src_dict[start_time], dst_dict[start_time], rel_type[start_time],
                                      rel_type_inverse[start_time], num_nodes)
        graph = utils.move_dgl_to_cuda(graph)
        model.train()
        if epoch == args.max_epochs:
            break
        epoch += 1
        loss_epoch = 0
        t0 = time.time()
        stop_time = valid_data[0][3]
        last_t = -1
        for batch_data in utils.make_train_batch(train_data, args.batch_size):
            batch_data = torch.from_numpy(batch_data).long()
            if use_cuda:
                batch_data = batch_data.cuda()
            time_list = batch_data[:, 3]
            unique_time_list, cnt_list = torch.unique(time_list, return_counts=True)
            cnt_list = cnt_list.cpu().numpy()
            cnt_list = cnt_list.tolist()
            # split batch data with time
            for b_data in torch.split(batch_data, cnt_list):
                # negative sample
                length = b_data.size()[0]  # length of b_data
                ones = torch.ones(length)
                zeros = torch.zeros(2 * length * args.negative_ratio)
                labels = torch.cat((ones, zeros), dim=0).cuda()
                negative_triples = utils.generate_negative_triples(b_data, length, args.negative_ratio, num_nodes)
                # current time
                current_t = b_data[0][3]
                # historical vocabulary
                sub_d_dict_er2e = all_his_dict_sub_er2e[current_t.item()]
                ob_d_dict_er2e = all_his_dict_ob_er2e[current_t.item()]
                # history relation evolution
                t_evolve_r = evolve_r[current_t.item()]
                t_len_r = len_r[current_t.item()]
                # update information
                if last_t < current_t:
                    edge_index = all_edge_index[current_t.item()]
                    edge_type = all_edge_type[current_t.item()]
                    edge_index = torch.LongTensor(edge_index).cuda().t()
                    edge_type = torch.LongTensor(edge_type).cuda()
                    model.reset_edge_information(edge_index, edge_type)
                    if last_t > -1:
                        graph = utils.update_graph(graph, src_dict[current_t.item()], dst_dict[current_t.item()],
                                                   rel_type[current_t.item()],
                                                   rel_type_inverse[current_t.item()],
                                                   rel_num_dict[last_t.item()])
                    last_t = current_t
                    c_win = current_t // time_unit
                    entity_embeds, rel_embeds = model.get_embeddings()
                    if c_win > args.seq_len:
                        c_win = args.seq_len
                        for timestamp in range(c_win - 2):
                            cache[timestamp] = cache[timestamp + 1]
                        cache[c_win - 2] = entity_embeds
                    else:
                        cache[c_win - 2] = entity_embeds
                look_up = cache.clone().detach().requires_grad_(False)
                # compute loss
                loss = model(b_data,
                             negative_triples,
                             labels,
                             current_t,
                             time_unit,
                             look_up,
                             t_evolve_r,
                             t_len_r,
                             sub_d_dict_er2e,
                             ob_d_dict_er2e,
                             graph)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += loss.item()
        t3 = time.time()
        print("Epoch {:04d} | Loss {:.4f} | time {:.4f}".
              format(epoch, loss_epoch / (len(train_data) / args.batch_size), t3 - t0))

        if epoch % args.valid_every == 0 and args.use_valid:
            print("start valid data evaluate...")
            model.eval()
            total_ranks = np.array([])
            for i in range(len(valid_data)):
                batch_data = valid_data[i]
                model.latest_time = valid_data[i][3]
                current_t = model.latest_time
                edge_index = all_edge_index[current_t.item()]
                edge_type = all_edge_type[current_t.item()]
                edge_index = torch.LongTensor(edge_index).cuda().t()
                edge_type = torch.LongTensor(edge_type).cuda()
                if last_t < current_t:
                    last_t = current_t
                    entity_embeds, rel_embeds = model.get_embeddings()
                    for timestamp in range(args.seq_len - 2):
                        cache[timestamp] = cache[timestamp + 1]
                    cache[args.seq_len - 2] = entity_embeds
                look_up = cache.clone().detach().requires_grad_(False)
                sub_d_dict_er2e = all_his_dict_sub_er2e[current_t.item()]
                ob_d_dict_er2e = all_his_dict_ob_er2e[current_t.item()]
                entity_pair = entity_pair_dict
                t_evolve_r = evolve_r[current_t.item()]
                t_len_r = len_r[current_t.item()]
                if use_cuda:
                    batch_data = batch_data.cuda()
                with torch.no_grad():
                    ranks = model.evaluate_filter(batch_data, stop_time, time_unit, total_data,
                                                                        sub_d_dict_er2e,
                                                                        ob_d_dict_er2e,
                                                                        edge_index,
                                                                        edge_type,
                                                                        look_up,
                                                                        entity_pair,
                                                                        t_evolve_r,
                                                                        t_len_r,
                                                                        graph
                                                                        )
                    total_ranks = np.concatenate((total_ranks, ranks))
                    model.latest_time = valid_data[i][3]
            mrr = np.mean(1.0 / total_ranks)
            mr = np.mean(total_ranks)
            hits = []
            for hit in [1, 3, 10]:
                avg_count = np.mean((total_ranks <= hit))
                hits.append(avg_count)
                print("valid Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
            print("valid MRR (filtered): {:.6f}".format(mrr))
            print("valid MR (filtered): {:.6f}".format(mr))
            if mrr > best_mrr:
                print("epoch {:04d} has achieved the best result".format(epoch))
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           save_model)

        if epoch % args.test_every == 0:
            print("start test data evaluate...")
            model.eval()
            total_ranks = np.array([])
            if args.load_model:
                checkpoint = torch.load(save_model, map_location=lambda storage, loc: storage)
                model.load_state_dict(checkpoint['state_dict'])
            for i in range(len(test_data)):
                batch_data = test_data[i]
                model.latest_time = test_data[i][3]
                current_t = model.latest_time
                edge_index = all_edge_index[current_t.item()]
                edge_type = all_edge_type[current_t.item()]
                edge_index = torch.LongTensor(edge_index).cuda().t()
                edge_type = torch.LongTensor(edge_type).cuda()
                if last_t < current_t:
                    last_t = current_t
                    entity_embeds, rel_embeds = model.get_embeddings()
                    for timestamp in range(args.seq_len - 2):
                        cache[timestamp] = cache[timestamp + 1]
                    cache[args.seq_len - 2] = entity_embeds
                look_up = cache.clone().detach().requires_grad_(False)
                sub_d_dict_er2e = all_his_dict_sub_er2e[current_t.item()]
                ob_d_dict_er2e = all_his_dict_ob_er2e[current_t.item()]
                entity_pair = entity_pair_dict
                t_evolve_r = evolve_r[current_t.item()]
                t_len_r = len_r[current_t.item()]
                if use_cuda:
                    batch_data = batch_data.cuda()
                with torch.no_grad():
                    ranks = model.evaluate_filter(batch_data, stop_time, time_unit, total_data,
                                                                        sub_d_dict_er2e,
                                                                        ob_d_dict_er2e,
                                                                        edge_index,
                                                                        edge_type,
                                                                        look_up,
                                                                        entity_pair,
                                                                        t_evolve_r,
                                                                        t_len_r,
                                                                        graph)
                    total_ranks = np.concatenate((total_ranks, ranks))
            mrr = np.mean(1.0 / total_ranks)
            mr = np.mean(total_ranks)
            hits = []
            for hit in [1, 3, 10]:
                avg_count = np.mean((total_ranks <= hit))
                hits.append(avg_count)
                print("test Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
            print("test MRR (filtered): {:.6f}".format(mrr))
            print("test MR (filtered): {:.6f}".format(mr))
    print("training done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIPN')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("-d", "--dataset", type=str, default='YAGO',
                        help="dataset to use")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="maximum epochs")
    parser.add_argument("--mini-epochs", type=int, default=1,
                        help="maximum epochs")
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k", type=int, default=1000,
                        help="cuttoff position")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--maxpool", type=int, default=1)
    parser.add_argument('--backup', action='store_true')
    parser.add_argument("--valid-every", type=int, default=10)
    parser.add_argument("--test-every", type=int, default=2)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--multi-step', type=bool, default=False)
    parser.add_argument('--used', type=bool, default=True)
    parser.add_argument('--use-valid', type=bool, default=False)
    parser.add_argument('--load-model', type=bool, default=False)
    '''comp_gcn'''
    # compgcn_transe compgcn_distmult compgcn_conve
    parser.add_argument('-comp-decoder', type=str, default='compgcn_distmult')
    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    # 'corr' 'sub' 'mult'
    parser.add_argument('-opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')
    parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim', dest='init_dim', default=200, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', dest='embed_dim', default=200, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=2, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-input_drop', dest='input_drop', default=0.3, type=float, help='Input dropout')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument('-score_func', dest='score_func', default='distmult', help='Score Function for Link prediction')
    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    args = parser.parse_args()
    print(args)
    train(args)
