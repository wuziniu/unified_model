import numpy as np
import random


def get_tree_lstm_batch_data(batch_id, seq, directory):
    target_cost_batch = np.load(directory+'/target_cost_'+str(batch_id)+'.np.npy')
    target_cardinality_batch = np.load(directory+'/target_cardinality_'+str(batch_id)+'.np.npy')
    operators_batch = np.load(directory+'/operators_'+str(batch_id)+'.np.npy')
    extra_infos_batch = np.load(directory+'/extra_infos_'+str(batch_id)+'.np.npy')
    condition1s_batch = np.load(directory+'/condition1s_'+str(batch_id)+'.np.npy')
    condition2s_batch = np.load(directory+'/condition2s_'+str(batch_id)+'.np.npy')
    samples_batch = np.load(directory+'/samples_'+str(batch_id)+'.np.npy')
    condition_masks_batch = np.load(directory+'/condition_masks_'+str(batch_id)+'.np.npy')

    if seq:
        mapping_batch = np.load(directory+'/position_encoding_'+str(batch_id)+'.np.npy')
    else:
        mapping_batch = np.load(directory+'/mapping_'+str(batch_id)+'.np.npy')
    return target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch,\
           condition2s_batch, samples_batch, condition_masks_batch, mapping_batch


def get_trans_batch_data(batch_id, seq, directory):
    target_cost_batch = np.load(directory + '/target_cost_'+ str(batch_id) + '.np.npy')
    target_cardinality_batch = np.load(directory + '/target_cardinality_' + str(batch_id) + '.np.npy')
    join_order_truth = np.load(directory+'/join_order_'+str(batch_id)+'.np.npy')
    trans_target = np.load(directory+'/trans_target_'+str(batch_id)+'.np.npy')
    operators_batch = np.load(directory+'/operators_'+str(batch_id)+'.np.npy')
    extra_infos_batch = np.load(directory+'/extra_infos_'+str(batch_id)+'.np.npy')
    condition1s_batch = np.load(directory+'/condition1s_'+str(batch_id)+'.np.npy')
    condition2s_batch = np.load(directory+'/condition2s_'+str(batch_id)+'.np.npy')
    samples_batch = np.load(directory+'/samples_'+str(batch_id)+'.np.npy')
    condition_masks_batch = np.load(directory+'/condition_masks_'+str(batch_id)+'.np.npy')
    leaf_node_marker = np.load(directory+'/leaf_node_marker_'+str(batch_id)+'.np.npy')
    res_mask = np.load(directory+'/res_mask_'+str(batch_id)+'.np.npy')
    adj_matrix = np.load(directory+'/join_order/adj_matrix_'+str(batch_id)+'.np.npy')

    if seq:
        mapping_batch = np.load(directory+'/position_encoding_'+str(batch_id)+'.np.npy')
    else:
        mapping_batch = np.load(directory+'/mapping_'+str(batch_id)+'.np.npy')
    return join_order_truth, target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, \
           condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch, leaf_node_marker, \
           trans_target, res_mask, adj_matrix


def get_batch_meta_learner_iterator(db_list, shuffle, seed, suffix, batch_num, test, directory="/mnt/train_data/meta_learner"):
    tuples = []
    random.seed(seed)

    for db_id in db_list:
        for batch_id in range(batch_num):
            tuples.append((db_id, batch_id))

    if shuffle:
        random.shuffle(tuples)

    prefix = "test_data" if test else "train_data"

    for db_id, batch_id in tuples:
        ground_truth_batch = np.load(f"{directory}/DB{db_id}/{prefix}{suffix}/ground_truth_{batch_id}.npy", allow_pickle=True)
        agg_matrix_batch = np.load(f"{directory}/DB{db_id}/{prefix}{suffix}/agg_matrix_{batch_id}.npy", allow_pickle=True)
        attn_mask_batch = np.load(f"{directory}/DB{db_id}/{prefix}{suffix}/attn_mask_{batch_id}.npy", allow_pickle=True)
        trans_target_batch = np.load(f"{directory}/DB{db_id}/{prefix}{suffix}/trans_target_{batch_id}.npy", allow_pickle=True)
        feature_encoding_batch = np.load(f"{directory}/DB{db_id}/{prefix}{suffix}/feature_encoding_{batch_id}.npy", allow_pickle=True)
        res_mask_batch = np.load(f"{directory}/DB{db_id}/{prefix}{suffix}/res_mask_{batch_id}.npy", allow_pickle=True)
        adj_matrix_batch = np.load(f"{directory}/DB{db_id}/{prefix}{suffix}/adj_matrix_{batch_id}.npy", allow_pickle=True)
        yield (ground_truth_batch, agg_matrix_batch, attn_mask_batch, trans_target_batch, feature_encoding_batch,
               res_mask_batch, adj_matrix_batch)