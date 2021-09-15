from tools.pytool import *
from tools.utils import *
from get_data_parameters import *
from representation_model import *
import torch
import time
import os
import argparse
import tqdm
from vector_loader import *


def train_lstm(train_start, train_end, validate_start, validate_end, num_epochs, patience, lr, data_parameters,
               model_parameters, directory, cuda, each_node, test, save_model):
    """
        Training tree-lstm on both cradinality estimation and cost estimation. 
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = data_parameters.condition_op_dim
    hidden_dim = model_parameters.trans_input_size
    hid_dim = model_parameters.cond_hidden_size
    middle_result_dim = 128
    task_num = 2
    if cuda:
        model = TreeLSTM(input_dim, hidden_dim, hid_dim, middle_result_dim, task_num).cuda()
    else:
        model = TreeLSTM(input_dim, hidden_dim, hid_dim, middle_result_dim, task_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='model/' + save_model + ".pt")

    model.train()
    start = time.time()
    vali_card_qerror_list = []
    vali_cost_qerror_list = []

    if test:
        num_epochs = 1
    for epoch in range(num_epochs):
        cost_loss_total = 0.
        card_loss_total = 0.
        model.train()
        print("=====================================")
        for batch_idx in tqdm.tqdm(range(train_start, train_end)):
            if test:
                break
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, \
            condition_maskss, mapping = get_tree_lstm_batch_data(batch_idx, 0, directory=directory)

            target_cost = torch.FloatTensor(target_cost).to(DEVICE)
            target_cardinality = torch.FloatTensor(target_cardinality).to(DEVICE)
            operatorss = torch.FloatTensor(operatorss).to(DEVICE).squeeze(0)
            extra_infoss = torch.FloatTensor(extra_infoss).to(DEVICE).squeeze(0)
            condition1ss = torch.FloatTensor(condition1ss).to(DEVICE).squeeze(0)
            condition2ss = torch.FloatTensor(condition2ss).to(DEVICE).squeeze(0)
            sampless = torch.FloatTensor(sampless).to(DEVICE).squeeze(0)
            condition_maskss = torch.FloatTensor(condition_maskss).to(DEVICE).squeeze(0).unsqueeze(2)
            mapping = torch.FloatTensor(mapping).to(DEVICE).squeeze(0)

            optimizer.zero_grad()
            estimate_cost, estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
                                                        condition_maskss, mapping)
            cost_loss, cost_loss_median, cost_loss_max = qerror_loss_each_node(estimate_cost, target_cost,
                                                                               data_parameters.cost_label_min,
                                                                               data_parameters.cost_label_max,
                                                                               mapping, each_node)
            card_loss, card_loss_median, card_loss_max = qerror_loss_each_node(estimate_cardinality,
                                                                               target_cardinality,
                                                                               data_parameters.card_label_min,
                                                                               data_parameters.card_label_max,
                                                                               mapping, each_node)
            # print(card_loss.item(), card_loss_median.item(), card_loss_max.item(), card_max_idx.item())
            loss = cost_loss + card_loss
            cost_loss_total += cost_loss.detach().cpu().item()
            card_loss_total += card_loss.detach().cpu().item()
            loss.backward()
            optimizer.step()
            # print('Training batch time: ', end - start)
        batch_num = train_end - train_start
        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total / batch_num,
                                                                                card_loss_total / batch_num))
        cost_loss_total_mean = 0.
        card_loss_total_mean = 0.

        cost_loss_total_median = 0.
        card_loss_total_median = 0.

        cost_loss_total_max = 0.
        card_loss_total_max = 0.
        model.eval()
        if test:
            validate_start, validate_end = 0, 1
            print("Test:")
        for batch_idx in range(validate_start, validate_end):
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, \
            condition_maskss, mapping = get_tree_lstm_batch_data(batch_idx, 0, directory=directory)
            
            target_cost = torch.FloatTensor(target_cost).to(DEVICE)
            target_cardinality = torch.FloatTensor(target_cardinality).to(DEVICE)
            operatorss = torch.FloatTensor(operatorss).to(DEVICE).squeeze(0)
            extra_infoss = torch.FloatTensor(extra_infoss).to(DEVICE).squeeze(0)
            condition1ss = torch.FloatTensor(condition1ss).to(DEVICE).squeeze(0)
            condition2ss = torch.FloatTensor(condition2ss).to(DEVICE).squeeze(0)
            sampless = torch.FloatTensor(sampless).to(DEVICE).squeeze(0)
            condition_maskss = torch.FloatTensor(condition_maskss).to(DEVICE).squeeze(0).unsqueeze(2)
            mapping = torch.FloatTensor(mapping).to(DEVICE).squeeze(0)

            estimate_cost, estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
                                                        condition_maskss, mapping)

            cost_loss, cost_loss_median, cost_loss_max = qerror_loss_each_node(estimate_cost, target_cost,
                                                                                data_parameters.cost_label_min,
                                                                                data_parameters.cost_label_max,
                                                                                mapping, each_node)
            card_loss, card_loss_median, card_loss_max = qerror_loss_each_node(estimate_cardinality,
                                                                                target_cardinality,
                                                                                data_parameters.card_label_min,
                                                                                data_parameters.card_label_max,
                                                                                mapping, each_node)
            # print(card_loss.item(), card_loss_median.item(), card_loss_max.item())
            cost_loss_total_mean += cost_loss.detach().cpu().item()
            card_loss_total_mean += card_loss.detach().cpu().item()

            cost_loss_total_median += cost_loss_median.detach().cpu().item()
            card_loss_total_median += card_loss_median.detach().cpu().item()

            cost_loss_total_max += cost_loss_max.detach().cpu().item()
            card_loss_total_max += card_loss_max.detach().cpu().item()

        batch_num = validate_end - validate_start
        early_stopping((cost_loss_total_mean+card_loss_total_mean) / batch_num, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

        vali_cost_qerror_list.append(cost_loss_total_mean / batch_num)
        vali_card_qerror_list.append(card_loss_total_mean / batch_num)
        print("=> Epoch {}, Validating results:".format(epoch))
        print("MEAN:  cost q-error: {}, card q-error: {}".format(cost_loss_total_mean / batch_num,
                                                                 card_loss_total_mean / batch_num))
        print("MEDIAN:  cost q-error: {}, card q-error: {}".format(cost_loss_total_median / batch_num,
                                                                   card_loss_total_median / batch_num))
        print("MAX:  cost q-error: {}, card q-error: {}".format(cost_loss_total_max / batch_num,
                                                                card_loss_total_max / batch_num))
    end = time.time()
    print("===========================================================================")
    print(f'total time cost:{end - start}')
    plot(vali_card_qerror_list, vali_cost_qerror_list, num_epochs)
    return model


def train_all_task(train_start, train_end, validate_start, validate_end, num_epochs, data_parameters, model_parameters, directory, patience, lr, card_task, cost_task, join_task,
                   each_node, test, f, save_model):
    """
        Multi-task training on cardinality estimation, cost estimation and join order selection. 
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    decay_rate = 0.9
    decay_point = 5
    model = UnifiedModel(model_parameters).to(DEVICE) 
    if test:
        print("Loading model:")
        checkpoint_finetune = torch.load('model/jo_checkpoint_bswc_sl.pt') 
        model_dict = model.state_dict()  # state_dict()
        pretrained_dict = {k: v for k, v in checkpoint_finetune.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    batch_size = 64
    teacher_forcing = TeacherForcing(start_tf=1, decay_rate=decay_rate, decay_point=decay_point, cur_epoch=0, end_epoch=50,
                                     verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'model/{save_model}.pt')
    criterion = nn.CrossEntropyLoss()
    model.train()
    start = time.time()
    max_corr_num = 0
    if test:
        num_epochs = 1
    for epoch in range(num_epochs):
        train_join_order_token_list = []
        train_join_order_seq_list = []
        train_cost_list = []
        train_card_list = []
        train_all_list = []
        print("=====================================")
        train_start_time = time.time()
        for batch_idx in tqdm.tqdm(range(train_start, train_end)):
            if test:
                break
            join_order_truth, cost_truth, card_truth, operatorss, extra_infoss, condition1ss, condition2ss, \
            sampless, condition_maskss, position_encoding, leaf_node_marker, trans_target, res_mask, adj_matrix = \
                get_trans_batch_data(batch_idx, 1, directory=directory)

            join_order_truth, cost_truth, card_truth, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, \
            position_encoding = torch.LongTensor(join_order_truth).to(DEVICE), torch.FloatTensor(cost_truth).to(DEVICE), \
                                torch.FloatTensor(card_truth).to(DEVICE), torch.FloatTensor(operatorss).to(DEVICE), \
                                torch.FloatTensor(extra_infoss).to(DEVICE), torch.FloatTensor(condition1ss).to(DEVICE), \
                                torch.FloatTensor(condition2ss).to(DEVICE), torch.FloatTensor(sampless).to(DEVICE), \
                                torch.FloatTensor(condition_maskss).to(DEVICE), torch.FloatTensor(position_encoding).to(DEVICE)
            trans_target = torch.FloatTensor(trans_target).to(DEVICE)
            leaf_node_marker = torch.LongTensor(leaf_node_marker).to(DEVICE)
            res_mask = torch.FloatTensor(res_mask).to(DEVICE)
            adj_matrix = torch.BoolTensor(adj_matrix).to(DEVICE)

            optimizer.zero_grad()
            cost_pred, card_pred, join_order_pred = model(operatorss, extra_infoss, condition1ss, condition2ss,
                                                          sampless, condition_maskss, position_encoding,
                                                          leaf_node_marker, 0, teacher_forcing, trans_target, res_mask,
                                                          adj_matrix)

            train_join_order_loss_token = join_order_loss(join_order_pred, join_order_truth, criterion)
            train_join_order_loss_seq = model.calculate_sequential_loss(operatorss, extra_infoss, condition1ss,
                                                                        condition2ss, sampless,
                                                                        condition_maskss, position_encoding,
                                                                        leaf_node_marker, trans_target,
                                                                        res_mask, adj_matrix, join_order_truth)
            if cost_task:
                train_cost_loss, _, _ = qerror_loss_seq_each_node(cost_pred, cost_truth, data_parameters.cost_label_min,
                                                                  data_parameters.cost_label_max, each_node)
            else:
                train_cost_loss = 0.
            if card_task:
                train_card_loss, _, _ = qerror_loss_seq_each_node(card_pred, card_truth, data_parameters.card_label_min,
                                                                  data_parameters.card_label_max, each_node)
            else:
                train_card_loss = 0.

            if not join_task:
                train_join_order_loss_token, train_join_order_loss_seq = 0, 0
            # dynamic_w.update(train_cost_loss, train_card_loss)
            tasks_loss = train_join_order_loss_token + train_card_loss + train_cost_loss \
                         + train_join_order_loss_seq

            train_join_order_token_list.append(train_join_order_loss_token)
            train_join_order_seq_list.append(train_join_order_loss_seq)
            train_cost_list.append(train_cost_loss)
            train_card_list.append(train_card_loss)
            train_all_list.append(tasks_loss)
            tasks_loss.backward()
            optimizer.step()

        batch_num = train_end - train_start
        train_end_time = time.time()
        teacher_forcing.check()
        scheduler.step(sum(train_all_list) / batch_num)
        print('Training batch time: ', train_end_time - train_start_time)
        print("Epoch {}, training all tasks loss: {}".format(epoch, sum(train_all_list) / batch_num))
        print("          training join order token loss: {}".format(sum(train_join_order_token_list) / batch_num))
        print("          training join order seq loss: {}".format(sum(train_join_order_seq_list) / batch_num))
        print("          training cost loss: {}".format(sum(train_cost_list) / batch_num))
        print("          training card loss: {}".format(sum(train_card_list) / batch_num))

        val_join_order_list = []
        val_cost_list = []
        val_cost_median_list = []
        val_cost_max_list = []

        val_card_list = []
        val_card_median_list = []
        val_card_max_list = []

        val_total_list = []
        joeu_list = []
        if not test:
            print("Validation: ")
        else:
            print("Test: ")
        cpl_correct_cnt = 0
        icpl_correct_cnt = 0
        pos_correct_cnt = 0
        total_pos_cnt = 0
        ills = 0
        model.eval()
        if test:
            validate_start, validate_end = 0, 1
        for batch_idx in tqdm.tqdm(range(validate_start, validate_end)):
            # print('Test_batch_idx: ', batch_idx)
            join_order_truth, cost_truth, card_truth, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, \
            position_encoding, leaf_node_marker, trans_target, res_mask, adj_matrix = get_trans_batch_data(batch_idx, 1,
                                                                                                          directory=directory)

            join_order_truth, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, \
            position_encoding = torch.LongTensor(join_order_truth).to(DEVICE), torch.FloatTensor(operatorss).to(DEVICE), \
                                torch.FloatTensor(extra_infoss).to(DEVICE), torch.FloatTensor(condition1ss).to(DEVICE), \
                                torch.FloatTensor(condition2ss).to(DEVICE), torch.FloatTensor(sampless).to(DEVICE), \
                                torch.FloatTensor(condition_maskss).to(DEVICE), torch.FloatTensor(position_encoding).to(DEVICE)
            leaf_node_marker = torch.LongTensor(leaf_node_marker).to(DEVICE)
            trans_target = torch.FloatTensor(trans_target).to(DEVICE)
            res_mask = torch.FloatTensor(res_mask).to(DEVICE)
            cost_truth = torch.FloatTensor(cost_truth).to(DEVICE)
            card_truth = torch.FloatTensor(card_truth).to(DEVICE)
            adj_matrix = torch.BoolTensor(adj_matrix).to(DEVICE)

            # cost_pred, card_pred, join_order_pred = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
            #                                 condition_maskss, position_encoding, leaf_node_marker, 1,
            #                                 teacher_forcing, trans_target, res_mask, adj_matrix)
            # c, p, t, ic = beam_search_prediction_compare(join_order_pred, join_order_truth)
            # joeu_list.append(0)

            cost_pred, card_pred, _ = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
                                            condition_maskss, position_encoding, leaf_node_marker, 0,
                                            teacher_forcing, trans_target, res_mask, adj_matrix)

            if join_task:
                join_order_pred, joeu_mean = model.beam_search_test(operatorss, extra_infoss, condition1ss,
                                                                         condition2ss, sampless, condition_maskss,
                                                                         position_encoding, leaf_node_marker,
                                                                         trans_target, res_mask, adj_matrix,
                                                                         join_order_truth)
                joeu_mean = joeu_mean.item()
                c, p, t, ic = beam_search_prediction_compare(join_order_pred, join_order_truth)
            else:
                join_order_pred, joeu_mean = 0, 0
                c, p, t, ic = 0, 0, 0, 0
            joeu_list.append(joeu_mean)

            cpl_correct_cnt += c
            pos_correct_cnt += p
            total_pos_cnt += t
            icpl_correct_cnt += ic
            val_loss = 0
            # val_loss = join_order_loss(join_order_pred, join_order_truth, res_mask, criterion)
            if cost_task:
                val_cost_loss, val_cost_median, val_cost_max = \
                    qerror_loss_seq_each_node(cost_pred, cost_truth, data_parameters.cost_label_min,
                                              data_parameters.cost_label_max, each_node)
                val_cost_loss, val_cost_median, val_cost_max = val_cost_loss.item(), val_cost_median.item(),\
                                                               val_cost_max.item()
            else:
                val_cost_loss, val_cost_median, val_cost_max = 0, 0, 0
            if card_task:
                val_card_loss, val_card_median, val_card_max = \
                    qerror_loss_seq_each_node(card_pred, card_truth, data_parameters.card_label_min,
                                              data_parameters.card_label_max, each_node)
                val_card_loss, val_card_median, val_card_max = val_card_loss.item(), val_card_median.item(), \
                                                               val_card_max.item()
            else:
                val_card_loss, val_card_median, val_card_max = 0, 0, 0
            tasks_loss = val_card_loss + val_cost_loss
            val_join_order_list.append(val_loss)
            val_cost_list.append(val_cost_loss)
            val_cost_median_list.append(val_cost_median)
            val_cost_max_list.append(val_cost_max)

            val_card_list.append(val_card_loss)
            val_card_median_list.append(val_card_median)
            val_card_max_list.append(val_card_max)

            val_total_list.append(tasks_loss)

        batch_num = validate_end - validate_start
        avg_loss = round(sum(val_total_list) / batch_num, 3)
        cur_corr_num = cpl_correct_cnt + icpl_correct_cnt
        if not test and join_task:
            early_stopping(-sum(joeu_list) / batch_num, model)
        else:
            early_stopping(avg_loss, model)
        max_corr_num = max(max_corr_num, cur_corr_num)

        cost_mean = round(sum(val_cost_list) / batch_num, 3)
        cost_median = round(sum(val_cost_median_list)/batch_num, 3)
        cost_max = round(sum(val_cost_max_list)/batch_num, 3)

        card_mean = round(sum(val_card_list) / batch_num, 3)
        card_median = round(sum(val_card_median_list) / batch_num, 3)
        card_max = round(sum(val_card_max_list) / batch_num, 3)
        print("Epoch {}, validation all tasks loss: {}".format(epoch, avg_loss))
        print("          validation join order loss: {}".format(sum(val_join_order_list) / batch_num))
        if test:
            f.write(f"test cost loss: {cost_mean}, median: {cost_median}, max: {cost_max}\n")
            f.write(f"test card loss: {card_mean}, median: {card_median}, max: {card_max}\n")
            f.write(f"Current the maximum of ccc+ic: {max_corr_num}\n")
            f.write(f"JoEU: {sum(joeu_list) / batch_num}\n")
            f.write(f"Illegal num: {ills}\n")
            print("**************************************************************************")
        print(f"         validation cost loss: {cost_mean}, median: {cost_median}, max: {cost_max}")
        print(f"         validation card loss: {card_mean}, median: {card_median}, max: {card_max}")
        print(f"         Complete correct count: {cpl_correct_cnt}/{batch_num * batch_size}")
        print(f"         Position correct count: {pos_correct_cnt}/{total_pos_cnt}")
        print(f"         Incomplete correct count: {icpl_correct_cnt}/{batch_num * batch_size}")
        print(f"         JoEU: {sum(joeu_list) / batch_num}")
        print(f"         Illegal num: {ills}")
        print(f"         Current the maximum of ccc+ic: {max_corr_num}")

        if not test and early_stopping.early_stop:
            print('Early stopping')
            break
    end = time.time()
    print("======================================")
    print(f'total time cost:{end - start}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="multi-task", help="multi_task: our model, tree-lstm: previous SOTA")
    parser.add_argument('--epochs', type=int, default=200, help="the training epochs")
    parser.add_argument('--patience', type=int, default=10, help="the patience of early stopping")
    parser.add_argument('--lr', type=float, default=1e-4, help="the learning rate")
    parser.add_argument('--bw', type=int, default=3, help="the beam width")
    parser.add_argument('--tis', type=int, default=128, help="the input size of transformer")
    parser.add_argument('--phs', type=int, default=256, help="the hidden size of predicate encoding")
    parser.add_argument('--hn', type=int, default=4, help="the head number of transformer")
    parser.add_argument('--cln', type=int, default=3, help="the layer number of (en/de)coder")
    parser.add_argument('--tdf', type=int, default=64, help="the size of transformer feedforward network")
    parser.add_argument('--card', type=int, default=1, help="the cardinality estimation button in multi-task learning")
    parser.add_argument('--cost', type=int, default=0, help="the cost estimation button in multi-task learning")
    parser.add_argument('--join', type=int, default=0, help="the join order selection button in multi-task learning")
    parser.add_argument('--en', type=int, default=0, help="whether considering the cardinality and cost of every node"
                                                          " in planing tree")
    parser.add_argument('--train_data_path', type=str, default="/mnt/train_data/join_order_each_node",
                        help="the path of train data")
    parser.add_argument('--test_data_path', type=str, default="/mnt/test_data/job",
                        help="the path of test data")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    mode = args.mode
    SEED = 666
    torch.manual_seed(SEED)

    start_time = time.time()
    data_parameters = get_data_parameters(mode)
    end_time = time.time()
    print(f"Loading data cost time: {end_time-start_time}")

    # train setting:
    epochs = args.epochs
    patience = args.patience
    lr = args.lr
    teacher_forcing_set = 1
    beam_width = args.bw

    # model set:
    input_size = data_parameters.condition_op_dim
    trans_input_size = args.tis
    cond_hidden_size = args.phs
    head_num = args.hn
    coder_layer_num = args.cln
    plan_pos_size = 18  # tree2dfs
    trans_dim_feedforward = args.tdf
    max_leaves_num = 7
    table_num = 21
    pos_flag = 1
    attn_flag = 0

    model_parameters = ModelParameters(input_size, trans_input_size, cond_hidden_size, head_num,
                                       coder_layer_num, plan_pos_size, pos_flag, attn_flag,
                                       max_leaves_num, trans_dim_feedforward, beam_width, table_num)

    print("******************************************************")
    print(f'learning rate: {lr}')
    print(f'trans_input_size: {trans_input_size}')
    print(f'head_num: {head_num}')
    print(f'coder_layer_num: {coder_layer_num}')
    print(f'cond_hidden_size: {cond_hidden_size}')
    print(f'trans_dim_feedforward: {trans_dim_feedforward}')
    print(f'beam_width: {beam_width}')
    print("******************************************************")

    if mode == "multi-task":
        data_path = args.train_data_path
        total_files_num = 100
        train_start = 0
        train_end = int(total_files_num*0.9)
        val_start = int(total_files_num*0.9)
        val_end = total_files_num
        cost_flag = args.cost
        card_flag = args.card
        join_flag = args.join
        each_node = args.en
        print(f"cost: {cost_flag}, cardinality: {card_flag}, join order: {join_flag}")
        save_model = card_flag*"Card" + cost_flag*"Cost" + join_flag*"Join" + "Trans"
        with open("joint_reuslt_log.txt", "w") as f:
            for test in [0, 1]:
                if test:
                    data_path = args.test_data_path
                train_all_task(train_start, train_end, val_start, val_end, epochs, data_parameters, model_parameters,
                               data_path, patience, lr, card_flag, cost_flag, join_flag, each_node, test, f, save_model)
    elif mode == "tree-lstm":
        data_path = args.train_data_path
        total_files_num = 295
        train_start = 0
        train_end = int(total_files_num*0.9)
        val_start = int(total_files_num*0.9)
        val_end = total_files_num
        each_node = 0
        save_model = "TreeLSTM"
        for test in [0, 1]:
            if test:
                data_path = args.test_data_path
            train_lstm(train_start, train_end, val_start, val_end, epochs, patience, lr, data_parameters, 
                       model_parameters, data_path, mode, each_node, test, save_model)