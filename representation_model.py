import torch
import torch.nn as nn
import torch.nn.functional as F
from get_data_parameters import *
import math
from copy import deepcopy


class TreeNode(object):
    def __init__(self, current_vec, parent):
        self.item = current_vec
        self.parent = parent
        self.children = []

    def get_parent(self):
        return self.parent

    def get_item(self):
        return self.item

    def get_children(self):
        return self.children

    def add_child(self, child):
        self.children.append(child)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, head_num, head_size):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.head_size = head_size

        self.input_dim = input_dim
        self.out_dim = head_num * head_size
        self.WQ = nn.Linear(self.input_dim, self.out_dim)
        self.WK = nn.Linear(self.input_dim, self.out_dim)
        self.WV = nn.Linear(self.input_dim, self.out_dim)

    def forward(self, seq_input):
        batch_size, seq_len, embedding_size = seq_input.size()
        Q = self.WQ(seq_input).view(batch_size, seq_len, self.head_num, self.head_size)
        Q = Q.permute(0, 2, 1, 3)  # bs, hn, sl, hs
        K = self.WK(seq_input).reshape(batch_size, seq_len, self.head_num, self.head_size)
        K = K.permute(0, 2, 1, 3)
        V = self.WV(seq_input).reshape(batch_size, seq_len, self.head_num, self.head_size)
        V = V.permute(0, 2, 1, 3)
        sim = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_size ** 0.5  # bs, hn, sl, sl
        attn = torch.matmul(F.softmax(sim, dim=3), V)  # bs, hn, sl, hs
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.out_dim)
        return attn


class TreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, hid_dim, middle_result_dim, task_num):
        #  parameters.condition_op_dim, 128, 256
        super(TreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        # The linear layer that maps from hidden state space to tag space

        self.sample_mlp = nn.Linear(1000, hid_dim)
        self.condition_mlp = nn.Linear(hidden_dim, hid_dim)
        #         self.out_mlp1 = nn.Linear(hidden_dim, middle_result_dim)
        #         self.hid_mlp1 = nn.Linear(15+108+2*hid_dim, hid_dim)
        #         self.out_mlp1 = nn.Linear(hid_dim, middle_result_dim)

        self.lstm2 = nn.LSTM(15 + 108 + 2 * hid_dim, hidden_dim, batch_first=True)

        #         self.lstm2_binary = nn.LSTM(15+108+2*hid_dim, hidden_dim, batch_first=True)
        #         self.lstm2_binary = nn.LSTM(15+108+2*hid_dim, hidden_dim, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hid_mlp2_task1 = nn.Linear(hidden_dim, hid_dim)
        self.hid_mlp2_task2 = nn.Linear(hidden_dim, hid_dim)
        self.batch_norm3 = nn.BatchNorm1d(hid_dim)
        self.hid_mlp3_task1 = nn.Linear(hid_dim, hid_dim)
        self.hid_mlp3_task2 = nn.Linear(hid_dim, hid_dim)
        self.out_mlp2_task1 = nn.Linear(hid_dim, 1)
        self.out_mlp2_task2 = nn.Linear(hid_dim, 1)

    #         self.hidden2values2 = nn.Linear(hidden_dim, action_num)

    def init_hidden(self, hidden_dim, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, hidden_dim).cuda(),
                torch.zeros(1, batch_size, hidden_dim).cuda())

    def predict(self, hidden_vec):
        out = self.batch_norm2(hidden_vec)

        out_task1 = F.relu(self.hid_mlp2_task1(out))
        out_task1 = self.batch_norm3(out_task1)
        out_task1 = F.relu(self.hid_mlp3_task1(out_task1))
        out_task1 = self.out_mlp2_task1(out_task1)
        out_task1 = torch.sigmoid(out_task1)

        out_task2 = F.relu(self.hid_mlp2_task2(out))
        out_task2 = self.batch_norm3(out_task2)
        out_task2 = F.relu(self.hid_mlp3_task2(out_task2))
        out_task2 = self.out_mlp2_task2(out_task2)
        out_task2 = torch.sigmoid(out_task2)
        return out_task1, out_task2

    def forward(self, operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping):
        num_level = condition1s.size()[0]  # get number of layer
        num_node_per_level = condition1s.size()[1]  # get maximum width
        num_condition_per_node = condition1s.size()[2]
        condition_op_length = condition1s.size()[3]  # get the dimension of condition_op

        inputs = condition1s.view(num_level * num_node_per_level, num_condition_per_node, condition_op_length)
        hidden = self.init_hidden(self.hidden_dim, num_level * num_node_per_level)

        out, hid = self.lstm1(inputs, hidden)
        last_output1 = hid[0].view(num_level * num_node_per_level, -1)  # (14*133, 128)

        # condition2
        num_level = condition2s.size()[0]
        num_node_per_level = condition2s.size()[1]
        num_condition_per_node = condition2s.size()[2]
        condition_op_length = condition2s.size()[3]

        inputs = condition2s.view(num_level * num_node_per_level, num_condition_per_node, condition_op_length)
        hidden = self.init_hidden(self.hidden_dim, num_level * num_node_per_level)

        out, hid = self.lstm1(inputs, hidden)
        last_output2 = hid[0].view(num_level * num_node_per_level, -1)  # (14*133, 128)

        last_output1 = F.relu(self.condition_mlp(last_output1))
        last_output2 = F.relu(self.condition_mlp(last_output2))
        last_output = (last_output1 + last_output2) / 2
        last_output = self.batch_norm1(last_output).view(num_level, num_node_per_level, -1)

        #         print (last_output.size())
        #         torch.Size([14, 133, 256])

        sample_output = F.relu(self.sample_mlp(samples))
        sample_output = sample_output * condition_masks

        out = torch.cat((operators, extra_infos, last_output, sample_output), 2)
        #         print (out.size())
        #         torch.Size([14, 133, 635])
        #         out = out * node_masks
        # start = time.time()
        hidden = self.init_hidden(self.hidden_dim, num_node_per_level)
        last_level = out[num_level - 1].view(num_node_per_level, 1, -1)
        #         torch.Size([133, 1, 635])
        _, (hid, cid) = self.lstm2(last_level, hidden)
        mapping = mapping.long()
        cost_pred = torch.zeros(size=(num_level, num_node_per_level, 1)).cuda()
        card_pred = torch.zeros(size=(num_level, num_node_per_level, 1)).cuda()
        for idx in reversed(range(0, num_level - 1)):
            cur_cost, cur_card = self.predict(hid[0])
            cost_pred[idx+1] = cur_cost
            card_pred[idx+1] = cur_card
            mapp_left = mapping[idx][:, 0]
            mapp_right = mapping[idx][:, 1]
            pad = torch.zeros_like(hid)[:, 0].unsqueeze(1).cuda()
            next_hid = torch.cat((pad, hid), 1)
            pad = torch.zeros_like(cid)[:, 0].unsqueeze(1).cuda()
            next_cid = torch.cat((pad, cid), 1)
            hid_left = torch.index_select(next_hid, 1, mapp_left)  # (133, 1, 128)
            cid_left = torch.index_select(next_cid, 1, mapp_left)
            hid_right = torch.index_select(next_hid, 1, mapp_right)  # (133, 1, 128)
            cid_right = torch.index_select(next_cid, 1, mapp_right)
            hid = (hid_left + hid_right) / 2
            cid = (cid_left + cid_right) / 2
            last_level = out[idx].view(num_node_per_level, 1, -1)
            _, (hid, cid) = self.lstm2(last_level, (hid, cid))
        cur_cost, cur_card = self.predict(hid[0])
        cost_pred[0] = cur_cost
        card_pred[0] = cur_card
        return cost_pred, card_pred


class UnifiedModel(nn.Module):
    # train: beam search with constraint + sequential loss + cost + cardinality
    def __init__(self, model_parameters: ModelParameters):
        #  parameters.condition_op_dim, 128, 256
        super(UnifiedModel, self).__init__()
        p = model_parameters
        self.pos_flag = p.pos_flag
        self.attn_flag = p.attn_flag
        self.hidden_size = p.trans_input_size
        self.trans_dim_feedforward = p.trans_dim_feedforward
        self.max_leaves_num = p.max_leaves_num
        self.beam_width = p.beam_width

        self.trans_input1 = nn.Linear(p.input_size, p.trans_input_size)
        self.trans_encoder1 = nn.TransformerEncoderLayer(d_model=p.trans_input_size,
                                                         nhead=p.head_num,
                                                         dim_feedforward=self.trans_dim_feedforward)
        self.trans_encoder1s = nn.TransformerEncoder(self.trans_encoder1, p.coder_layer_num)

        self.batch_norm1 = nn.BatchNorm1d(p.cond_hidden_size)

        self.start_vec = nn.Parameter(torch.FloatTensor(size=(1, self.hidden_size)))
        self.start_vec = nn.init.kaiming_normal_(self.start_vec, a=0, mode='fan_in')
        self.sample_mlp = nn.Linear(1000, p.cond_hidden_size)
        self.condition_mlp = nn.Linear(p.trans_input_size, p.cond_hidden_size)

        self.trans_input = nn.Linear(p.cond_hidden_size + p.cond_hidden_size + 123, p.trans_input_size)
        self.plan_tree_pos_embedding = nn.Linear(p.plan_pos_size, p.trans_input_size)

        self.plan_trad_pos_encoding = PositionalEncoding(d_model=p.trans_input_size)
        self.trans_encoder2 = nn.TransformerEncoderLayer(d_model=p.trans_input_size,
                                                         nhead=p.head_num,
                                                         dim_feedforward=self.trans_dim_feedforward)
        self.trans_encoder2s = nn.TransformerEncoder(self.trans_encoder2, p.coder_layer_num)

        self.batch_norm2 = nn.BatchNorm1d(p.trans_input_size)

        # attention
        self.plan_W0 = nn.Linear(p.trans_input_size, 1)
        self.plan_W1 = nn.Linear(p.trans_input_size, p.trans_input_size)
        self.plan_W2 = nn.Linear(p.trans_input_size, p.trans_input_size)
        self.hid_mlp2_task1 = nn.Linear(p.trans_input_size, p.cond_hidden_size)
        self.hid_mlp2_task2 = nn.Linear(p.trans_input_size, p.cond_hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(p.cond_hidden_size)
        self.hid_mlp3_task1 = nn.Linear(p.cond_hidden_size, p.cond_hidden_size)
        self.hid_mlp3_task2 = nn.Linear(p.cond_hidden_size, p.cond_hidden_size)
        self.out_mlp2_task1 = nn.Linear(p.cond_hidden_size, 1)
        self.out_mlp2_task2 = nn.Linear(p.cond_hidden_size, 1)

        self.trans_decoder = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=p.head_num,
                                                        dim_feedforward=self.trans_dim_feedforward)
        self.trans_decoders = nn.TransformerDecoder(self.trans_decoder, p.coder_layer_num)
        self.output = nn.Linear(p.trans_input_size, self.max_leaves_num)
        self.embedding = nn.Embedding(self.max_leaves_num, self.hidden_size)


    def encode(self, operators, extra_infos, condition1s, condition2s, samples, condition_masks, plan_pos_encoding,
               leaf_node_marker):
        batch_size, nodes_num, num_condition_per_node, condition_op_length = condition1s.size()
        batch_num = batch_size * nodes_num

        inputs = condition1s.view(batch_num, num_condition_per_node, condition_op_length)
        inputs = self.trans_input1(inputs)

        hid = self.trans_encoder1s(inputs.permute(1, 0, 2))  # (10, 64*25, 128)
        last_output1 = torch.mean(hid, dim=0).view(batch_num, -1)  # (64*25, 128)

        inputs = condition2s.view(batch_num, num_condition_per_node, condition_op_length)
        inputs = self.trans_input1(inputs)
        hid = self.trans_encoder1s(inputs.permute(1, 0, 2))
        last_output2 = torch.mean(hid, dim=0).view(batch_num, -1)  # (64*25, 128)

        last_output1 = F.relu(self.condition_mlp(last_output1))
        last_output2 = F.relu(self.condition_mlp(last_output2))
        last_output = (last_output1 + last_output2) / 2  # ?
        last_output = self.batch_norm1(last_output).view(batch_size, nodes_num, -1)

        #         print (last_output.size())
        #         torch.Size([64, 25, 256])

        sample_output = F.relu(self.sample_mlp(samples))
        sample_output = sample_output * condition_masks

        out = torch.cat((operators, extra_infos, last_output, sample_output), 2)
        #         print (out.size())
        #         torch.Size([64, 25, 635])
        #         out = out * node_masks
        out1 = self.trans_input(out)  # (64, 25, 128)

        out_jo = torch.cat((torch.zeros(size=(batch_size, nodes_num, 15)).cuda(), out[:, :, 15:]), dim=2)
        out2 = self.trans_input(out_jo)  # (64, 25, 128)

        none_padding = torch.zeros(size=(batch_size, 1, self.hidden_size)).cuda()
        plan_rep_cat = torch.cat((none_padding, out2), dim=1).view(batch_size * (nodes_num + 1), -1)  # (64, 26, 128)
        leaf_node_rep = torch.index_select(plan_rep_cat, 0, leaf_node_marker).view(batch_size, self.max_leaves_num, -1)

        if self.pos_flag == 1:
            plan_pos_encoding = self.plan_tree_pos_embedding(plan_pos_encoding)  # (64, 25, 128)
            out1 = out1 + plan_pos_encoding
        elif self.pos_flag == 2:
            out1 = self.plan_trad_pos_encoding(out1)

        cost_card_rep = self.trans_encoder2(out1.permute(1, 0, 2)).permute(1, 0, 2)
        join_order_rep = self.trans_encoder2(leaf_node_rep.permute(1, 0, 2)).permute(1, 0, 2)  # (64, 7, 128)
        return cost_card_rep, join_order_rep

    def decode(self, join_order_rep, cost_card_rep, test, teacher_forcing, trans_target, res_mask, adj_matrix):
        batch_size, nodes_num, _ = cost_card_rep.shape
        trans_target = trans_target.permute(1, 0, 2)  # (7, 64, 128)
        #######################  join order  ############################
        if not test:
            jo_output = torch.ones(size=(batch_size, 1, self.max_leaves_num)).cuda()
            tgt_pre = self.start_vec.expand(batch_size, self.hidden_size).unsqueeze(0)
            trans_target = torch.cat((tgt_pre, trans_target), dim=0)  # (8, 64, 128) 拼了初始向量，一点问题都没有
            for seq_id in range(join_order_rep.size()[1]):
                if torch.rand(1).cuda()[0] < teacher_forcing.tf:
                    tgt_pre = self.trans_decoders(tgt=trans_target[seq_id].unsqueeze(0),
                                                  memory=join_order_rep.permute(1, 0, 2))
                else:
                    tgt_pre = self.trans_decoders(tgt=tgt_pre, memory=join_order_rep.permute(1, 0, 2))
                res = F.relu(self.output(tgt_pre.permute(1, 0, 2)))  # (64, 1, 7)
                max_index = torch.argmax(res, dim=2)  # (64, 1)
                one_hot = torch.zeros(batch_size, self.max_leaves_num).cuda().scatter_(1, max_index, 1)  # (64, 7)
                tgt_pre = torch.cat((one_hot, trans_target[seq_id + 1, :, 7:]), dim=1).unsqueeze(0)  # (1, 64, 128)
                # tgt_pre = torch.cat((one_hot, torch.zeros(batch_size, 121).cuda()), dim=1).unsqueeze(0)
                jo_output = torch.cat((jo_output, res), dim=1)
            join_order_output = jo_output[:, 1:, :]  # (64, 7, 7)
        else:
            tgt_pre = self.start_vec.expand(batch_size, self.hidden_size).unsqueeze(0)
            jo_output = torch.ones(size=(batch_size, 1, self.max_leaves_num)).cuda()
            for seq_id in range(join_order_rep.size()[1]):
                tgt_pre = self.trans_decoders(tgt=tgt_pre, memory=join_order_rep.permute(1, 0, 2))
                res = F.relu(self.output(tgt_pre.permute(1, 0, 2)))
                res += res_mask.unsqueeze(1)
                max_index = torch.argmax(res, dim=2)  # (64, 1)
                for bi in range(batch_size):
                    res_mask[bi, max_index[bi]] = -float("inf")
                one_hot = torch.zeros(batch_size, self.max_leaves_num).cuda().scatter_(1, max_index, 1)  # (64, 7)
                tgt_pre = torch.cat((one_hot, trans_target[seq_id, :, 7:]), dim=1).unsqueeze(0)  # (1, 64, 128)
                # tgt_pre = torch.cat((one_hot, torch.zeros(batch_size, 121).cuda()), dim=1).unsqueeze(0)
                jo_output = torch.cat((jo_output, res), dim=1)
            join_order_output = torch.argmax(jo_output[:, 1:, :], dim=2)  # (64, 7, 7)

        #             # check_legality
        #             for batch_idx in range(batch_size):
        #                 if not self.check_legality(join_order_output[batch_idx], adj_matrix[batch_idx]):

        #################### cardinality estimation cost estiamtion #####################
        plan_rep = cost_card_rep.reshape(batch_size * nodes_num, -1)
        plan_rep = self.batch_norm2(plan_rep)

        out_task1 = F.relu(self.hid_mlp2_task1(plan_rep))
        out_task1 = self.batch_norm3(out_task1)
        out_task1 = F.relu(self.hid_mlp3_task1(out_task1))
        out_task1 = self.out_mlp2_task1(out_task1)
        cost_output = torch.sigmoid(out_task1)
        cost_output = cost_output.reshape(batch_size, nodes_num, -1)

        out_task2 = F.relu(self.hid_mlp2_task2(plan_rep))
        out_task2 = self.batch_norm3(out_task2)
        out_task2 = F.relu(self.hid_mlp3_task2(out_task2))
        out_task2 = self.out_mlp2_task2(out_task2)
        card_output = torch.sigmoid(out_task2)
        card_output = card_output.reshape(batch_size, nodes_num, -1)

        return cost_output, card_output, join_order_output

    def forward(self, operators, extra_infos, condition1s, condition2s, samples, condition_masks, plan_pos_encoding,
                leaf_node_marker, test, teacher_forcing, trans_target, res_mask, adj_matrix):
        cost_card_rep, join_order_rep = self.encode(operators, extra_infos, condition1s, condition2s, samples,
                                                    condition_masks, plan_pos_encoding, leaf_node_marker)
        cost_output, card_output, join_order_output = self.decode(join_order_rep, cost_card_rep, test, teacher_forcing,
                                                                  trans_target, res_mask, adj_matrix)
        batch_size = cost_output.shape[0]
        res_mask = res_mask.unsqueeze(2).expand(batch_size, self.max_leaves_num, self.max_leaves_num)
        if not test:
            join_order_output += res_mask
        return cost_output, card_output, join_order_output

    def step(self, tgt_input, join_order_rep, res_mask):
        tgt_output = self.trans_decoders(tgt=tgt_input, memory=join_order_rep.permute(1, 0, 2))
        tgt_output = F.leaky_relu(self.output(tgt_output.permute(1, 0, 2)))
        tgt_output = tgt_output.squeeze(1)
        tgt_output += res_mask
        return tgt_output

    def generate_mask(self, res_mask, top_k):
        # (64, 7)
        cur_num, k = top_k.shape
        res_mask = res_mask.unsqueeze(1).repeat(1, k, 1)
        for i in range(cur_num):
            for j in range(k):
                res_mask[i, j, top_k[i, j]] = -float("inf")
        return res_mask.reshape(-1, self.max_leaves_num)

    def beam_search_test(self, operators, extra_infos, condition1s, condition2s, samples, condition_masks,
                         plan_pos_encoding, leaf_node_marker, trans_target, res_mask_truth, adj_matrix, ground_truth):
        # adj_matrix (64, 7, 7)
        res_mask = deepcopy(res_mask_truth)
        batch_size = res_mask.shape[0]
        _, join_order_rep = self.encode(operators, extra_infos, condition1s, condition2s, samples, condition_masks,
                                        plan_pos_encoding, leaf_node_marker)

        first_input = self.start_vec
        #         join_order_output = torch.ones((1, self.max_leaves_num, self.max_leaves_num)).cuda()
        results = []
        joeus = 0
        for batch_id in range(batch_size):
            res_index = self.generate_results_test(first_input, join_order_rep[batch_id], trans_target[batch_id],
                                                   res_mask[batch_id], adj_matrix[batch_id])
            res_index = res_index.long()
            # print(res_index)
            # if not self.check_legality(res_index, adj_matrix[batch_id]):
            #     print("???????????????")
            table_num = res_index.shape[0]
            joeu = self.calculate_joeu(res_index.unsqueeze(0), ground_truth[batch_id])
            joeus += joeu
            res_index = torch.cat((res_index, -1 * torch.ones(self.max_leaves_num - table_num).cuda())).unsqueeze(0)
        #             print(res_index)
            results.append(res_index)
        results = torch.cat(results)
        return results, joeus / batch_size

    def test_mask(self, seen, valid_tables):
        # (seen, )  (1, )  (7, 7)
        res_mask = torch.zeros(self.max_leaves_num).cuda()
        for idx in range(seen.shape[0]):
            res_mask[seen[idx]] = -float("inf")
        for idx in range(self.max_leaves_num):
            if res_mask[idx] == -float("inf"):
                continue
            if valid_tables[idx] == 0:
                res_mask[idx] = -float("inf")
        return res_mask

    def generate_results_test(self, input_vec, join_order_rep, trans_target, res_mask, adj_matrix):
        # (1, 128)  (7, 128)  (7, 128)
        res_mask = res_mask.unsqueeze(0)  # (1, 7)

        cur_output = self.step(input_vec.unsqueeze(0), join_order_rep.unsqueeze(1), res_mask)  # (7, )
        final_output = torch.ones(1, 1).cuda()
        cur_prob = torch.ones(1).cuda()
        table_num = res_mask[0].shape[0] - torch.isinf(res_mask[0]).sum()
        record_table_num = table_num
        valid_tables = torch.zeros(1, 7).bool().cuda()
        # print(table_num, res_mask)
        for i in range(table_num):
            # time step
            tmp_dis = cur_output.view(-1, self.max_leaves_num)  # (1, 7)
            tmp_dis = F.softmax(tmp_dis, dim=1)
            cur_num = tmp_dis.shape[0]
            #             print(top_k_value, top_k)
            next_final_output = []
            next_cur_output = []
            next_prob = []
            next_res_mask = []
            next_valid_tables = []
            # print(f"time step {i}")
            for j in range(cur_num):
                # the possible number of last time step
                table_rest_num = res_mask[j].shape[0] - torch.isinf(res_mask[j]).sum()
                can_expand_num = min(self.beam_width, table_rest_num)
                # print(can_expand_num)
                if can_expand_num == 0:
                    continue
                top_k_value, top_k = tmp_dis[j].topk(can_expand_num)  # (1, 3)
                tmp_final_output = []
                for k in range(can_expand_num):
                    # current expand number
                    next_prob.append(cur_prob[j]*top_k_value[k])
                    tmp_valid_tables = valid_tables[j] | adj_matrix[top_k[k]]
                    seen = torch.cat((final_output[j], torch.Tensor([top_k[k]]).cuda()))
                    tmp_final_output.append(seen.unsqueeze(0))

                    tmp_res_mask = self.test_mask(seen[1:].long(), tmp_valid_tables)
                    # print(tmp_res_mask)
                    next_res_mask.append(tmp_res_mask.unsqueeze(0))
                    next_valid_tables.append(tmp_valid_tables.unsqueeze(0))

                    one_hot = F.one_hot(top_k[k], num_classes=self.max_leaves_num).float()  # (1, 7)
                    cur_tgt = torch.cat((one_hot, trans_target[i, self.max_leaves_num:])).unsqueeze(0)  # (1, 128)
                    tmp_cur_output = self.step(cur_tgt.unsqueeze(0), join_order_rep.unsqueeze(0), tmp_res_mask)  # (1, 7)
                    next_cur_output.append(tmp_cur_output)
                next_final_output.append(torch.cat(tmp_final_output))

            res_mask = torch.cat(next_res_mask)
            # print(res_mask.shape)
            valid_tables = torch.cat(next_valid_tables)
            # print(valid_tables.shape)
            final_output = torch.cat(next_final_output)
            # print(final_output.shape)
            cur_prob = torch.Tensor(next_prob)
            # print(cur_prob.shape)
            cur_output = torch.cat(next_cur_output)
            # print(cur_output.shape)

        final_output = final_output[:, 1:]
        assert final_output.shape[1] == record_table_num
        final_index = torch.argmax(cur_prob)  # (prob_num)

        return final_output[final_index]

    def generate_results(self, input_vec, join_order_rep, trans_target, res_mask, adj_matrix, ground_truth):
        # (1, 128)  (7, 128)  (7, 128)
        # beam search (train or test)
        final_prob = torch.ones(1).cuda()
        final_output = torch.ones(size=(1, 1, self.max_leaves_num)).cuda()
        final_index = torch.ones((1, 1)).cuda()
        res_mask = res_mask.unsqueeze(0)  # (1, 7)

        cur_output = self.step(input_vec.unsqueeze(0), join_order_rep.unsqueeze(1), res_mask)  # (7, )
        final_output = torch.cat((final_output, cur_output.view(1, 1, self.max_leaves_num)), dim=1)
        table_num = res_mask[0].shape[0] - torch.isinf(res_mask[0]).sum()

        for i in range(table_num):
            #             print(res_mask)
            table_rest_num = res_mask[0].shape[0] - torch.isinf(res_mask[0]).sum()

            tmp_dis = cur_output.view(-1, self.max_leaves_num)  # (1, 7)
            tmp_dis = F.softmax(tmp_dis, dim=1)
            cur_num = tmp_dis.shape[0]

            can_expand_num = min(self.beam_width, table_rest_num)
            top_k_value, top_k = tmp_dis.topk(can_expand_num, dim=1)  # (1, 3)
            #             print(top_k_value, top_k)

            final_prob = final_prob.unsqueeze(1).repeat(1, can_expand_num).reshape(-1)
            #             print(final_prob, top_k_value.reshape(-1))
            final_index = final_index.unsqueeze(1).repeat(1, can_expand_num, 1).reshape(cur_num * can_expand_num, -1)
            final_index = torch.cat((final_index, top_k.view(cur_num * can_expand_num, 1)), dim=1)
            final_prob = final_prob * top_k_value.reshape(-1)

            res_mask = self.generate_mask(res_mask, top_k)  # (3, 7)
            one_hot = F.one_hot(top_k, num_classes=self.max_leaves_num).float()  # (1, en, 7)
            cur_tgt = trans_target[i, :].unsqueeze(1).repeat(1, cur_num * can_expand_num, 1)
            cur_tgt = cur_tgt.reshape(cur_num, can_expand_num, self.hidden_size)
            cur_tgt = torch.cat((one_hot, cur_tgt[:, :, self.max_leaves_num:]), dim=2)  # (1, en, 128)
            cur_tgt = cur_tgt.view(-1, self.hidden_size).unsqueeze(0)

            join_order_rep_tmp = join_order_rep.unsqueeze(1).repeat(1, cur_num * can_expand_num, 1)  # (7, ., 128)
            cur_output = self.step(cur_tgt, join_order_rep_tmp, res_mask)  # (3, 7)

            final_output = final_output.unsqueeze(1).repeat(1, can_expand_num, 1, 1).reshape(cur_num * can_expand_num,
                                                                                             -1, self.max_leaves_num)
            cur_output = cur_output.view(-1, 1, self.max_leaves_num)
            final_output = torch.cat((final_output, cur_output), dim=1)
        final_output = final_output[:, 1:, :]   # (prob_num, table_num, 7)
        #         print(final_output.shape, final_prob.shape)
        #         print(f"final_prob {final_prob}")

        sort_prob, sort_prob_index = torch.sort(final_prob, descending=True)  # (prob_num)
        all_prob_index = final_index[:, 1:]  # (prob_num, table_num)
        illegal_index = torch.zeros((1, table_num)).cuda()
        illegal_prob = torch.zeros(1).cuda()

        legal_index = torch.zeros((1, table_num)).cuda()
        legal_prob = torch.zeros(1).cuda()

        for prob_idx in range(sort_prob_index.shape[0]):
            cur_index = all_prob_index[sort_prob_index[prob_idx]].long()
            cur_prob = sort_prob[prob_idx]
            if self.check_legality(cur_index, adj_matrix):
                legal_index = torch.cat((legal_index, cur_index.unsqueeze(0)), dim=0)
                legal_prob = torch.cat((legal_prob, cur_prob.unsqueeze(0)), dim=0)
            else:
                illegal_index = torch.cat((illegal_index, cur_index.unsqueeze(0)), dim=0)
                illegal_prob = torch.cat((illegal_prob, cur_prob.unsqueeze(0)), dim=0)

        illegal_index = illegal_index[1:]
        illegal_prob = illegal_prob[1:]
        legal_index = legal_index[1:]
        legal_prob = legal_prob[1:]

        if min(legal_prob.shape) == 0:
            optimal_prob = 0
            optimal_index = torch.zeros(1).long().cuda()
        else:
            joeus = self.calculate_joeu(legal_index, ground_truth)
            optimal_index = torch.argmax(joeus)
            optimal_prob = legal_prob[optimal_index]

        legal_index = self.del_tensor_ele(legal_index, optimal_index.item())
        legal_prob = self.del_tensor_ele(legal_prob, optimal_index.item())
        return illegal_index, illegal_prob, legal_index, legal_prob, optimal_index, optimal_prob

    def del_tensor_ele(self, tensor, index):
        return torch.cat((tensor[:index], tensor[index+1:]), dim=0)

    def calculate_sequential_loss(self, operators, extra_infos, condition1s, condition2s, samples, condition_masks,
                             plan_pos_encoding, leaf_node_marker, trans_target, res_mask_truth, adj_matrix, ground_truth):
        # adj_matrix (64, 7, 7)
        # beam search train seq loss
        # print("Calculating sequential loss...")
        res_mask = deepcopy(res_mask_truth)
        batch_size = res_mask.shape[0]
        _, join_order_rep = self.encode(operators, extra_infos, condition1s, condition2s, samples, condition_masks,
                                        plan_pos_encoding, leaf_node_marker)

        first_input = self.start_vec
        seq_loss = 0
        for batch_id in range(batch_size):
            illegal_index, illegal_prob, legal_index, legal_prob, optimal_index, optimal_prob \
                = self.generate_results(first_input, join_order_rep[batch_id], trans_target[batch_id],
                                        res_mask[batch_id], adj_matrix[batch_id], ground_truth[batch_id])
            table_num = illegal_index.shape[0]
            if min(illegal_prob.shape) == 0:  # it is reasonable that loss is 0 if there are no illeagal outputs
                illegal_loss = 0
            else:
                illegal_loss = torch.log(torch.sum(torch.exp(1/table_num * torch.log(illegal_prob))))
            if optimal_prob == 0:
                optimal_loss = 0
            else:
                optimal_loss = -torch.log(optimal_prob)
            if min(legal_prob.shape) == 0:
                legal_risk = 0
            else:
                joeus = self.calculate_joeu(legal_index, ground_truth[batch_id])
                legal_risk = self.calculate_risk(legal_prob, joeus)
                optimal_loss = -torch.log(optimal_prob)
            seq_loss += (optimal_loss + illegal_loss + legal_risk)
            return seq_loss/batch_size

    def calculate_joeu(self, legal_index, gt):
        # (prob, table_num)  (prob)  (7)
        if min(legal_index.shape) == 0:
            return 0
        prob_num, table_num = legal_index.shape
        joeus = torch.ones(1).cuda()
        for prob_id in range(prob_num):
            cnt = 0
            cur_index = legal_index[prob_id]
            if cur_index[0] == gt[0]:
                for t in range(1, table_num):
                    if cur_index[t] == gt[t]:
                        cnt += 1
                    else:
                        break
            elif cur_index[0] == gt[1] and cur_index[1] == gt[0]:
                cnt += 1
                for t in range(2, table_num):
                    if cur_index[t] == gt[t]:
                        cnt += 1
                    else:
                        break
            joeus = torch.cat((joeus, torch.Tensor([cnt/(table_num-1)]).cuda()), dim=0)
        return joeus[1:]

    def calculate_risk(self, legal_prob, joeus):
        return torch.sum((1 - joeus) * legal_prob)

    def check_legality(self, join_order_index, adj_matrix):
        # (table_num)   (7, 7)
        cur_tables_num = join_order_index.shape[0]
        # check
        seen = deepcopy(adj_matrix[join_order_index[0]]).bool()
        for idx in range(1, cur_tables_num):
            if seen[join_order_index[idx]] == 0:
                return False
            seen = seen | adj_matrix[join_order_index[idx]]
        return True


class MetaLearner(nn.Module):
    # transferability across databases
    # meta feature
    def __init__(self, trans_input_size, head_num, trans_dim_feedforward, coder_layer_num, max_leaves_num, drop_out):
        #  parameters.condition_op_dim, 128, 256
        super(MetaLearner, self).__init__()
        self.max_leaves_num = max_leaves_num
        self.head_num = head_num
        self.hidden_size = trans_input_size
        self.trans_encoder1 = nn.TransformerEncoderLayer(d_model=trans_input_size,
                                                         nhead=head_num,
                                                         dim_feedforward=trans_dim_feedforward,
                                                         dropout=drop_out)
        self.trans_encoder1s = nn.TransformerEncoder(self.trans_encoder1, coder_layer_num)

        self.start_vec = nn.Parameter(torch.FloatTensor(size=(1, self.hidden_size)))
        self.start_vec = nn.init.kaiming_normal_(self.start_vec, a=0, mode='fan_in')

        self.trans_encoder2 = nn.TransformerEncoderLayer(d_model=trans_input_size,
                                                         nhead=head_num,
                                                         dim_feedforward=trans_dim_feedforward)
        self.trans_encoder2s = nn.TransformerEncoder(self.trans_encoder2, coder_layer_num)

        self.trans_decoder = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=head_num,
                                                        dim_feedforward=trans_dim_feedforward, dropout=drop_out)
        self.trans_decoders = nn.TransformerDecoder(self.trans_decoder, coder_layer_num)
        self.output = nn.Linear(self.hidden_size, self.max_leaves_num)

    def encode(self, feature_encoding, encoder_mask, agg_matrix):
        # (64, 30, 128)  (64, 30, 30)  (64, 10, 30)
        # print(feature_encoding.shape, encoder_mask.shape, agg_matrix.shape)
        bsz, tgt_len, tgt_len = encoder_mask.shape
        encoder_mask = encoder_mask.unsqueeze(1).repeat(1, self.head_num, 1, 1).view(-1, tgt_len, tgt_len)
        assert self.head_num*bsz == encoder_mask.shape[0]
        join_order_rep = self.trans_encoder1s(feature_encoding.permute(1, 0, 2), mask=encoder_mask)  # (20, 64, 128)
        join_order_rep = torch.matmul(agg_matrix, join_order_rep.permute(1, 0, 2))
        return join_order_rep

    def decode(self, join_order_rep, test, teacher_forcing, trans_target, res_mask, adj_matrix):
        batch_size, nodes_num, _ = join_order_rep.shape
        # join_order_rep = torch.zeros_like(join_order_rep)
        trans_target = trans_target.squeeze()
        trans_target = trans_target.permute(1, 0, 2)  # (10, 64, 128)
        org_res_mask = deepcopy(res_mask)
        #######################  join order  ############################
        if not test:
            jo_output = torch.ones(size=(batch_size, 1, self.max_leaves_num)).cuda()
            tgt_pre = self.start_vec.expand(batch_size, self.hidden_size).unsqueeze(0)
            trans_target = torch.cat((tgt_pre, trans_target), dim=0)  # (8, 64, 128)
            for seq_id in range(join_order_rep.size()[1]):
                if torch.rand(1).cuda()[0] < teacher_forcing.tf:
                    tgt_pre = self.trans_decoders(tgt=trans_target[seq_id].unsqueeze(0),
                                                  memory=join_order_rep.permute(1, 0, 2))
                else:
                    tgt_pre = self.trans_decoders(tgt=tgt_pre, memory=join_order_rep.permute(1, 0, 2))
                res = torch.sigmoid(self.output(tgt_pre.permute(1, 0, 2)))  # (64, 1, 7)
                max_index = torch.argmax(res, dim=2)  # (64, 1)
                one_hot = torch.zeros(batch_size, self.max_leaves_num).cuda().scatter_(1, max_index, 1)  # (64, 7)
                tgt_pre = torch.cat((one_hot, trans_target[seq_id + 1, :, self.max_leaves_num:]), dim=1).unsqueeze(0)  # (1, 64, 128)
                jo_output = torch.cat((jo_output, res), dim=1)
            join_order_output = jo_output[:, 1:, :]  # (64, 7, 7)
        else:
            seen = torch.zeros((batch_size, 1)).cuda()
            valid_tables = torch.zeros((batch_size, self.max_leaves_num)).bool().cuda()
            tgt_pre = self.start_vec.expand(batch_size, self.hidden_size).unsqueeze(0)
            jo_output = torch.ones(size=(batch_size, 1, self.max_leaves_num)).cuda()
            for seq_id in range(join_order_rep.size()[1]):
                tgt_pre = self.trans_decoders(tgt=tgt_pre, memory=join_order_rep.permute(1, 0, 2))
                res = torch.sigmoid(self.output(tgt_pre.permute(1, 0, 2)))
                res += res_mask.unsqueeze(1)
                max_index = torch.argmax(res, dim=2)  # (64, 1)
                seen = torch.cat((seen, max_index), dim=1)

                for bi in range(batch_size):
                    valid_tables[bi] = valid_tables[bi] | adj_matrix[bi, max_index[bi]]
                    res_mask[bi] = org_res_mask[bi] + self.test_mask(seen[bi][1:].long(), valid_tables[bi])
                    if seq_id == 0 and res_mask[bi].min() == -float("inf") and res_mask[bi].max() == -float("inf"):
                        print(max_index[bi], adj_matrix[bi], valid_tables[bi])

                # tgt_pre = self.embedding(max_index.squeeze()).unsqueeze(0)
                one_hot = torch.zeros(batch_size, self.max_leaves_num).cuda().scatter_(1, max_index, 1)  # (64, 7)
                tgt_pre = torch.cat((one_hot, trans_target[seq_id, :, self.max_leaves_num:]), dim=1).unsqueeze(
                    0)  # (1, 64, 128)
                jo_output = torch.cat((jo_output, res), dim=1)
            join_order_output = jo_output[:, 1:, :]

        return join_order_output  # (64, 10, 10)

    def forward(self, feature_encoding, agg_matrix, encoder_mask, test, teacher_forcing, trans_target, res_mask, adj_matrix):
        join_order_rep = self.encode(feature_encoding, encoder_mask, agg_matrix)
        join_order_output = self.decode(join_order_rep, test, teacher_forcing, trans_target, res_mask, adj_matrix)
        batch_size = join_order_output.shape[0]
        res_mask = res_mask.squeeze().unsqueeze(1).expand(batch_size, self.max_leaves_num, self.max_leaves_num)
        if not test:
            join_order_output += res_mask
        return join_order_output

    def test_mask(self, seen, valid_tables):
        res_mask = torch.zeros(self.max_leaves_num).cuda()
        for idx in range(seen.shape[0]):
            res_mask[seen[idx]] = -float("inf")
        for idx in range(self.max_leaves_num):
            if res_mask[idx] == -float("inf"):
                continue
            if valid_tables[idx] == 0:
                res_mask[idx] = -float("inf")
        return res_mask
