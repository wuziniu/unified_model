import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def unnormalize(vecs, mini, maxi):
    return torch.exp(vecs * (maxi - mini) + mini)


def plot(vali_card_qerror_list, vali_cost_qerror_list, num_epochs):

    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(211)
    plt.xlabel("epoch")
    plt.ylabel("card_qerror")
    plt.grid(axis='y')
    for a, b in zip([i for i in range(num_epochs)], vali_card_qerror_list):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    plt.plot(vali_card_qerror_list, color='blue', linestyle='-', marker='o')

    ax2 = plt.subplot(212)
    plt.xlabel("epoch")
    plt.ylabel("cost_qerror")
    plt.grid(axis='y')
    for a, b in zip([i for i in range(num_epochs)], vali_cost_qerror_list):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    plt.plot(vali_cost_qerror_list, color='blue', linestyle='-', marker='o')
    plt.savefig('./test_pic.jpg')
    plt.show()


def qerror_loss(preds, targets, mini, maxi):
    qerror = []
    preds = unnormalize(preds, mini, maxi)  # 解归一化
    targets = unnormalize(targets, mini, maxi)
    for i in range(len(targets)):
        if preds[i] > targets[i]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror)), torch.median(torch.cat(qerror)), torch.max(torch.cat(qerror))


def qerror_loss_each_node(preds, targets, mini, maxi, mapping, validation):
    # (14, 133),  (14, 133), (14, 133, 2)
    qerror = []
    preds = unnormalize(preds, mini, maxi)  # 解归一化
    targets = unnormalize(targets, mini, maxi)
    if not validation:
        for level in range(len(mapping)-1):
            for w in range(len(mapping[0])):
                left, right = mapping[level][w]
                if left != 0:
                    qerror.append(max(preds[level+1][left-1], targets[level+1][left-1])/min(preds[level+1][left-1], targets[level+1][left-1]))
                if right != 0:
                    qerror.append(
                        max(preds[level+1][right-1], targets[level+1][right-1]) / min(preds[level+1][right-1], targets[level+1][right-1]))
    for i in range(64):
        qerror.append(max(preds[0][i], targets[0][i]) / min(preds[0][i], targets[0][i]))
    return torch.mean(torch.cat(qerror)), torch.median(torch.cat(qerror)), torch.max(torch.cat(qerror))


def qerror_loss_seq_each_node(preds, targets, mini, maxi, each_node):
    # (64, 25, 1),  (64, 25)
    qerror = []
    preds = unnormalize(preds, mini, maxi)  # 解归一化
    # print(preds[:, 0].cpu().detach().numpy().reshape(-1).tolist())
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)
    for i in range(len(targets)):
        for j in range(len(targets[0])):
            if targets[i][j] == -1:
                targets[i][j] = -float("inf")

    targets = unnormalize(targets, mini, maxi)
    if each_node:
        for batch_id in range(len(preds)):
            for i in range(len(preds[0])):
                if targets[batch_id][i] != 0:
                    qerror.append(max(preds[batch_id][i], targets[batch_id][i]) / min(preds[batch_id][i],
                                                                                      targets[batch_id][i]))
    else:
        for i in range(len(targets)):
            if targets[i][0] != -1:
                qerror.append(max(preds[i][0], targets[i][0]) / min(preds[i][0], targets[i][0]))
    return torch.mean(torch.cat(qerror)), torch.median(torch.cat(qerror)), torch.max(torch.cat(qerror))


def join_order_loss(pred, ground_truth, cel):
    loss_list = []
    batch_size, cur_batch_nodes, position_size = pred.shape
    for i in range(batch_size):
        pd_idx = 0
        while pd_idx < len(ground_truth[i]) and ground_truth[i][pd_idx] != -1:
            pd_idx += 1
        loss_list.append(cel(pred[i][:pd_idx], ground_truth[i][:pd_idx]))
    return sum(loss_list) / len(loss_list)


def beam_search_prediction_compare(pred, ground_truth):
    pred = pred.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    batch_size = pred.shape[0]
    complete_correct_cnt = 0
    position_correct_cnt = 0
    incomplete_correct_cnt = 0
    pd_cnt = 0
    for i in range(batch_size):
        pd_idx = 0
        while pd_idx < len(ground_truth[i]) and ground_truth[i][pd_idx] != -1:
            pd_idx += 1
        pred_idx_list = pred[i, :pd_idx]
        gt_idx_list = ground_truth[i, :pd_idx]
        assert len(set(pred_idx_list)) == pd_idx
        pd_cnt += pd_idx
        cur_corr = (pred_idx_list == gt_idx_list).sum()
        if pred_idx_list[0] == gt_idx_list[1] and cur_corr == pd_idx - 2:
            incomplete_correct_cnt += 1

        position_correct_cnt += cur_corr
        if cur_corr == pd_idx:
            complete_correct_cnt += 1
    return complete_correct_cnt, position_correct_cnt, pd_cnt, incomplete_correct_cnt


def gen_random_seq(num_table, adj_matrix):
    # (10, 10)
    unseen = [_ for _ in range(num_table)]
    start = random.choice(unseen)
    unseen.remove(start)
    res = [start]
    cur = adj_matrix[start]

    for i in range(len(unseen)):
        rd = random.choice(unseen)
        while cur[rd] == 0:
            rd = random.choice(unseen)
        res.append(rd)
        unseen.remove(rd)
        cur = cur | adj_matrix[rd]
    return res


def random_prediction(ground_truth, adj_matrix):
    num_tables = len(ground_truth[0])
    ground_truth = ground_truth.detach().cpu().numpy()
    adj_matrix = adj_matrix.detach().cpu().numpy()
    batch_size = ground_truth.shape[0]
    complete_correct_cnt = 0
    position_correct_cnt = 0
    incomplete_correct_cnt = 0
    pd_cnt = 0
    res_pred = []
    for i in range(batch_size):
        pd_idx = 0
        while pd_idx < len(ground_truth[i]) and ground_truth[i][pd_idx] != -1:
            pd_idx += 1
        pred_idx_list = gen_random_seq(pd_idx, adj_matrix[i])
        gt_idx_list = ground_truth[i, :pd_idx]

        assert set(pred_idx_list) == set(gt_idx_list)
        pred_idx_record = pred_idx_list + [-1 for _ in range(num_tables - pd_idx)]
        res_pred.append(pred_idx_record)
        # print(pred_idx_list, gt_idx_list)
        pd_cnt += pd_idx
        cur_corr = (pred_idx_list == gt_idx_list).sum()
        if pred_idx_list[0] == gt_idx_list[1] and cur_corr == pd_idx - 2:
            incomplete_correct_cnt += 1

        position_correct_cnt += cur_corr
        if cur_corr == pd_idx:
            complete_correct_cnt += 1
    return complete_correct_cnt, position_correct_cnt, pd_cnt, incomplete_correct_cnt, res_pred


def output_file(pred, ground_truth):
    # (64, 10, 10)  (64, 10)
    pred = pred.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

    out_list = []
    num_tables = len(ground_truth[0])
    pred = np.argmax(pred, axis=2)
    batch_size = pred.shape[0]
    for i in range(batch_size):
        pd_idx = 0
        while pd_idx < num_tables and ground_truth[i][pd_idx] != -1:
            pd_idx += 1
        pred_idx_list = pred[i, :pd_idx].tolist() + [-1 for _ in range(num_tables - pd_idx)]
        out_list.append(pred_idx_list)
    return out_list


def prediction_compare(pred, ground_truth):
    # (64, 10, 10)  (64, 10)
    pred = pred.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

    pred = np.argmax(pred, axis=2)
    batch_size = pred.shape[0]
    complete_correct_cnt = 0
    position_correct_cnt = 0
    incomplete_correct_cnt = 0
    pd_cnt = 0
    for i in range(batch_size):
        pd_idx = 0
        while pd_idx < len(ground_truth[i]) and ground_truth[i][pd_idx] != -1:
            pd_idx += 1
        pred_idx_list = pred[i, :pd_idx]
        gt_idx_list = ground_truth[i, :pd_idx]
        # print(pred_idx_list, gt_idx_list)
        assert set(pred_idx_list) == set(gt_idx_list)
        pd_cnt += pd_idx
        cur_corr = (pred_idx_list == gt_idx_list).sum()
        if pred_idx_list[0] == gt_idx_list[1] and cur_corr == pd_idx - 2:
            incomplete_correct_cnt += 1

        position_correct_cnt += cur_corr
        if cur_corr == pd_idx:
            complete_correct_cnt += 1
    return complete_correct_cnt, position_correct_cnt, pd_cnt, incomplete_correct_cnt

