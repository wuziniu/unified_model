from representation_model import *
from vector_loader import *
from tools.pytool import *
from tools.utils import *
import torch
import time
import pickle
import tqdm
import os
import argparse


def zero_shot_learning(train_start, train_end, validate_start, validate_end, num_epochs, patience, lr, trans_input_size,
                       head_num, trans_dim_feedforward, coder_layer_num, max_leaves_num, drop_out, train_db, val_db,
                       save_f, db_path, save_model):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    decay_rate = 0.9
    decay_point = 10
    batch_size = 64
    seed = 666
    model = MetaLearner(trans_input_size, head_num, trans_dim_feedforward, coder_layer_num, max_leaves_num, drop_out).to(DEVICE)
    teacher_forcing = TeacherForcing(start_tf=1, decay_rate=decay_rate, decay_point=decay_point, cur_epoch=0, end_epoch=50,
                                     verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'model/{save_model}.pt')
    criterion = nn.CrossEntropyLoss()
    model.train()
    start = time.time()
    max_corr_num = 0

    print(f"train_db: {train_db}")
    print(f"val_db: {val_db}")
    train_batches = 100
    val_batches = 20

    for epoch in range(num_epochs):
        # db_list, shuffle, seed, suffix, batch_num, test,
        train_iter = get_batch_meta_learner_iterator(train_db, 1, seed, 3, train_batches, 0, db_path)
        val_iter = get_batch_meta_learner_iterator(val_db, 0, seed, 3, val_batches, 0, db_path)
        train_loss = []
        print("=====================================")
        save_f.write("=====================================\n")
        train_start_time = time.time()
        for train_data in tqdm.tqdm(train_iter):
            test = 0
            ground_truth_batch, agg_matrix_batch, attn_mask_batch, trans_target_batch, feature_encoding_batch, \
            res_mask_batch, adj_matrix_batch = train_data

            ground_truth_batch = torch.LongTensor(ground_truth_batch).to(DEVICE)
            agg_matrix_batch = torch.Tensor(agg_matrix_batch).to(DEVICE)
            trans_target_batch = torch.FloatTensor(trans_target_batch).to(DEVICE)
            attn_mask_batch = torch.Tensor(attn_mask_batch).to(DEVICE)
            feature_encoding_batch = torch.FloatTensor(feature_encoding_batch).to(DEVICE)
            res_mask_batch = torch.FloatTensor(res_mask_batch).to(DEVICE)
            adj_matrix_batch = torch.BoolTensor(adj_matrix_batch).to(DEVICE)

            optimizer.zero_grad()
            join_order_output = model(feature_encoding_batch, agg_matrix_batch, attn_mask_batch, test, teacher_forcing,
                                      trans_target_batch, res_mask_batch, adj_matrix_batch)
            loss = join_order_loss(join_order_output, ground_truth_batch, criterion)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()

        batch_num = train_end - train_start
        train_end_time = time.time()
        teacher_forcing.check()
        print('Training batch time: ', train_end_time - train_start_time)
        print("Epoch {}, training all tasks loss: {}".format(epoch, sum(train_loss) / batch_num))
        save_f.write("Epoch {}, training all tasks loss: {}\n".format(epoch, sum(train_loss) / batch_num))

        cpl_correct_cnt, icpl_correct_cnt, pos_correct_cnt, total_pos_cnt = 0, 0, 0, 0
        cpl_correct_cnt_rp, pos_correct_cnt_rp, icpl_correct_cnt_rp = 0, 0, 0
        model.eval()
        results = []
        random_res = []

        for val_data in tqdm.tqdm(val_iter):
            test = 1
            ground_truth_batch, agg_matrix_batch, attn_mask_batch, trans_target_batch, feature_encoding_batch, \
            res_mask_batch, adj_matrix_batch = val_data
            ground_truth_batch = torch.LongTensor(ground_truth_batch).to(DEVICE)
            agg_matrix_batch = torch.Tensor(agg_matrix_batch).to(DEVICE)
            trans_target_batch = torch.FloatTensor(trans_target_batch).to(DEVICE)
            attn_mask_batch = torch.Tensor(attn_mask_batch).to(DEVICE)
            feature_encoding_batch = torch.FloatTensor(feature_encoding_batch).to(DEVICE)
            res_mask_batch = torch.FloatTensor(res_mask_batch).to(DEVICE)
            adj_matrix_batch = torch.BoolTensor(adj_matrix_batch).to(DEVICE)

            join_order_output = model(feature_encoding_batch, agg_matrix_batch, attn_mask_batch, test, teacher_forcing,
                                      trans_target_batch, res_mask_batch, adj_matrix_batch)
            results += output_file(join_order_output, ground_truth_batch)
            ccc, pcc, tpn, icc = prediction_compare(join_order_output, ground_truth_batch)
            ccc_rp, pcc_rp, _, icc_rp, rd_list = random_prediction(ground_truth_batch, adj_matrix_batch)
            random_res += rd_list
            cpl_correct_cnt += ccc
            pos_correct_cnt += pcc
            icpl_correct_cnt += icc
            total_pos_cnt += tpn

            cpl_correct_cnt_rp += ccc_rp
            icpl_correct_cnt_rp += icc_rp
            pos_correct_cnt_rp += pcc_rp

        batch_num = validate_end - validate_start
        cur_corr = cpl_correct_cnt+icpl_correct_cnt
        pickle.dump(results, open("/mnt/train_data/meta_learner/log/prediction_rd.pkl", "wb"))
        if cur_corr > max_corr_num:
            max_corr_num = cur_corr
            pickle.dump(results, open("/mnt/train_data/meta_learner/log/prediction.pkl", "wb"))
        early_stopping(-cpl_correct_cnt-icpl_correct_cnt, model)
        print(f"Epoch {epoch}: ")
        print(f"         Complete correct count: {cpl_correct_cnt}/{len(val_db)*val_batches * batch_size}")
        save_f.write(f"         Position correct count: {pos_correct_cnt}/{total_pos_cnt}\n")
        print(f"         Position correct count: {pos_correct_cnt}/{total_pos_cnt}")
        print(f"         Incomplete correct count: {icpl_correct_cnt}/{len(val_db)*val_batches * batch_size}")
        print(f"         Current the maximum of ccc+ic: {max_corr_num}")
        save_f.write(f"         Current the maximum of ccc+ic: {max_corr_num}\n")
        print(f"Random Prediction:")
        print(f"         Complete correct count:  {cpl_correct_cnt_rp}/{len(val_db)*val_batches * batch_size}")
        print(f"         Incomplete correct count: {icpl_correct_cnt_rp}/{len(val_db)*val_batches * batch_size}")

        if early_stopping.early_stop:
            print('Early stopping')
            break
    end = time.time()
    print("======================================")
    print(f'total time cost:{end - start}')
    return max_corr_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="multi-task", help="multi_task: our model, tree-lstm: previous SOTA")
    parser.add_argument('--epochs', type=int, default=200, help="the training epochs")
    parser.add_argument('--patience', type=int, default=10, help="the patience of early stopping")
    parser.add_argument('--lr', type=float, default=1e-6, help="the learning rate")
    parser.add_argument('--bw', type=int, default=3, help="the beam width")
    parser.add_argument('--phs', type=int, default=256, help="the hidden size of predicate encoding")
    parser.add_argument('--hn', type=int, default=4, help="the head number of transformer")
    parser.add_argument('--cln', type=int, default=3, help="the layer number of (en/de)coder")
    parser.add_argument('--tdf', type=int, default=64, help="the size of transformer feedforward network")
    parser.add_argument('--dp', type=int, default=0.2, help="the drop out of transformer")
    parser.add_argument('--train_db', type=int, nargs="+", help="the list of training databases")
    parser.add_argument('--test_db', type=int, nargs="+", help="the path of test data")
    parser.add_argument('--db_path', type=str, default="/mnt/train_data/meta_learner",
                        help="the directory of databases")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    SEED = 666
    torch.manual_seed(SEED)

    # train setting:
    epochs = args.epochs
    patience = args.patience
    lr = args.lr

    # model set:
    trans_input_size = 128
    head_num = args.hn
    coder_layer_num = args.cln
    trans_dim_feedforward = args.tdf
    drop_out = args.dp

    teacher_forcing_set = 1
    print("******************************************************")
    print(f'learning rate: {lr}')
    print(f'trans_input_size: {trans_input_size}')
    print(f'head_num: {head_num}')
    print(f'coder_layer_num: {coder_layer_num}')
    print(f'trans_dim_feedforward: {trans_dim_feedforward}')
    print(f'drop_out: {drop_out}')
    print("******************************************************")

    total_files_num = 120
    train_start = 0
    train_end = int(total_files_num*0.9)
    val_start = int(total_files_num*0.9)
    val_end = total_files_num

    db_path = args.db_path
    train_db = args.train_db
    val_db = args.test_db
    max_leaves_num = 10
    save_model = "meta_learner"
    with open(f"meta_learning_log.txt", "w") as save_f:
        save_f.write(f"train_db: {train_db}, val_db: {val_db}\n\n")

        t = zero_shot_learning(train_start, train_end, val_start, val_end, epochs, patience, lr, trans_input_size,
                               head_num, trans_dim_feedforward, coder_layer_num, max_leaves_num, drop_out,
                               train_db, val_db, save_f, db_path, save_model)

 