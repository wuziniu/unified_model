import pandas as pd
import pickle
import os
import json


class DataParameters:
    def __init__(self, condition_max_num, indexes_id, tables_id, columns_id, physic_ops_id, column_total_num,
                 table_total_num, index_total_num, physic_op_total_num, condition_op_dim, compare_ops_id, bool_ops_id,
                 bool_ops_total_num, compare_ops_total_num, data, min_max_column, word_vectors, cost_label_min,
                 cost_label_max, card_label_min, card_label_max):
        self.condition_max_num = condition_max_num
        self.indexes_id = indexes_id
        self.tables_id = tables_id
        self.columns_id = columns_id
        self.physic_ops_id = physic_ops_id
        self.column_total_num = column_total_num
        self.table_total_num = table_total_num
        self.index_total_num = index_total_num
        self.physic_op_total_num = physic_op_total_num
        self.condition_op_dim = condition_op_dim
        self.compare_ops_id = compare_ops_id
        self.bool_ops_id = bool_ops_id
        self.bool_ops_total_num = bool_ops_total_num
        self.compare_ops_total_num = compare_ops_total_num
        self.data = data
        self.min_max_column = min_max_column
        self.word_vectors = word_vectors
        self.cost_label_min = cost_label_min
        self.cost_label_max = cost_label_max
        self.card_label_min = card_label_min
        self.card_label_max = card_label_max


class ModelParameters:
    def __init__(self, input_size, trans_input_size, cond_hidden_size, head_num, coder_layer_num, plan_pos_size,
                 pos_flag, attn_flag, max_leaves_num, trans_dim_feedforward, beam_width, table_num):
        self.input_size = input_size
        self.trans_input_size = trans_input_size
        self.cond_hidden_size = cond_hidden_size
        self.head_num = head_num
        self.coder_layer_num = coder_layer_num
        self.plan_pos_size = plan_pos_size
        self.pos_flag = pos_flag
        self.attn_flag = attn_flag
        self.max_leaves_num = max_leaves_num
        self.trans_dim_feedforward = trans_dim_feedforward
        self.beam_width = beam_width
        self.table_num = table_num


def load_numeric_min_max(path):
    with open(path, 'r') as f:
        min_max_column = json.loads(f.read())
    return min_max_column


def get_data_parameters(mode):
    print("Preparing data...")
    dataset = load_dataset('table_data/')

    if os.path.exists('table_data/table_info.txt'):
        column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, table_names = \
            pickle.load(open('table_data/table_info.txt', 'rb'))
    else:
        column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, table_names = \
            prepare_dataset(dataset)
        pickle.dump(prepare_dataset(dataset), open('table_data/table_info.txt', 'wb'))
    print('Data prepared!')
    print("Loading min and max data...")
    min_max_column = load_numeric_min_max('table_data/min_max_vals.json')
    print('Min max loaded!')

    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    if mode == "numeric":
        condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num + 1
    else:
        condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num + 600

    # plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max \
    #     = obtain_upper_bound_query_size('../train_data/', True, mode)
    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = 72, 13, \
        -9.210340371976182, 14.411289201204804, -2.3025850929940455, 19.536825097210908
    cost_label_min_test, cost_label_max_test, card_label_min_test, card_label_max_test \
        = float("inf"), -float("inf"), float("inf"), -float("inf")
    # plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size('../test_files_open_source/plans_seq_sample.json')
    # plan_node_max_num_test, condition_max_num_test, cost_label_min_test, cost_label_max_test, card_label_min_test, card_label_max_test = obtain_upper_bound_query_size('../test_files_open_source/plans_seq_sample.json')
    cost_label_min = min(cost_label_min, cost_label_min_test)
    cost_label_max = max(cost_label_max, cost_label_max_test)
    card_label_min = min(card_label_min, card_label_min_test)
    card_label_max = max(card_label_max, card_label_max_test)
    print('query upper size prepared')

    data_parameters = DataParameters(condition_max_num, indexes_id, tables_id, columns_id, physic_ops_id,
                                     column_total_num, table_total_num, index_total_num, physic_op_total_num,
                                     condition_op_dim, compare_ops_id, bool_ops_id, bool_ops_total_num,
                                     compare_ops_total_num, dataset, min_max_column, [], cost_label_min,
                                     cost_label_max, card_label_min, card_label_max)

    return data_parameters


def load_dataset(dir_path):
    data = dict()
    data["aka_name"] = pd.read_csv(dir_path + '/aka_name.csv', header=None, low_memory=False)
    data["aka_title"] = pd.read_csv(dir_path + '/aka_title.csv', header=None, low_memory=False)
    data["cast_info"] = pd.read_csv(dir_path + '/cast_info.csv', header=None, low_memory=False)
    data["char_name"] = pd.read_csv(dir_path + '/char_name.csv', header=None, low_memory=False)
    data["company_name"] = pd.read_csv(dir_path + '/company_name.csv', header=None, low_memory=False)
    data["company_type"] = pd.read_csv(dir_path + '/company_type.csv', header=None, low_memory=False)
    data["comp_cast_type"] = pd.read_csv(dir_path + '/comp_cast_type.csv', header=None, low_memory=False)
    data["complete_cast"] = pd.read_csv(dir_path + '/complete_cast.csv', header=None, low_memory=False)
    data["info_type"] = pd.read_csv(dir_path + '/info_type.csv', header=None, low_memory=False)
    data["keyword"] = pd.read_csv(dir_path + '/keyword.csv', header=None, low_memory=False)
    data["kind_type"] = pd.read_csv(dir_path + '/kind_type.csv', header=None, low_memory=False)
    data["link_type"] = pd.read_csv(dir_path + '/link_type.csv', header=None, low_memory=False)
    data["movie_companies"] = pd.read_csv(dir_path + '/movie_companies.csv', header=None, low_memory=False)
    data["movie_info"] = pd.read_csv(dir_path + '/movie_info.csv', header=None, low_memory=False)
    data["movie_info_idx"] = pd.read_csv(dir_path + '/movie_info_idx.csv', header=None, low_memory=False)
    data["movie_keyword"] = pd.read_csv(dir_path + '/movie_keyword.csv', header=None, low_memory=False)
    data["movie_link"] = pd.read_csv(dir_path + '/movie_link.csv', header=None, low_memory=False)
    data["name"] = pd.read_csv(dir_path + '/name.csv', header=None, low_memory=False)
    data["person_info"] = pd.read_csv(dir_path + '/person_info.csv', header=None, low_memory=False)
    data["role_type"] = pd.read_csv(dir_path + '/role_type.csv', header=None, low_memory=False)
    data["title"] = pd.read_csv(dir_path + '/title.csv', header=None, low_memory=False)

    aka_name_column = {
        'id': 0,
        'person_id': 1,
        'name': 2,
        'imdb_index': 3,
        'name_pcode_cf': 4,
        'name_pcode_nf': 5,
        'surname_pcode': 6,
        'md5sum': 7
    }

    aka_title_column = {
        'id': 0,
        'movie_id': 1,
        'title': 2,
        'imdb_index': 3,
        'kind_id': 4,
        'production_year': 5,
        'phonetic_code': 6,
        'episode_of_id': 7,
        'season_nr': 8,
        'episode_nr': 9,
        'note': 10,
        'md5sum': 11
    }

    cast_info_column = {
        'id': 0,
        'person_id': 1,
        'movie_id': 2,
        'person_role_id': 3,
        'note': 4,
        'nr_order': 5,
        'role_id': 6
    }

    char_name_column = {
        'id': 0,
        'name': 1,
        'imdb_index': 2,
        'imdb_id': 3,
        'name_pcode_nf': 4,
        'surname_pcode': 5,
        'md5sum': 6
    }

    comp_cast_type_column = {
        'id': 0,
        'kind': 1
    }

    company_name_column = {
        'id': 0,
        'name': 1,
        'country_code': 2,
        'imdb_id': 3,
        'name_pcode_nf': 4,
        'name_pcode_sf': 5,
        'md5sum': 6
    }

    company_type_column = {
        'id': 0,
        'kind': 1
    }

    complete_cast_column = {
        'id': 0,
        'movie_id': 1,
        'subject_id': 2,
        'status_id': 3
    }

    info_type_column = {
        'id': 0,
        'info': 1
    }

    keyword_column = {
        'id': 0,
        'keyword': 1,
        'phonetic_code': 2
    }

    kind_type_column = {
        'id': 0,
        'kind': 1
    }

    link_type_column = {
        'id': 0,
        'link': 1
    }

    movie_companies_column = {
        'id': 0,
        'movie_id': 1,
        'company_id': 2,
        'company_type_id': 3,
        'note': 4
    }

    movie_info_idx_column = {
        'id': 0,
        'movie_id': 1,
        'info_type_id': 2,
        'info': 3,
        'note': 4
    }

    movie_keyword_column = {
        'id': 0,
        'movie_id': 1,
        'keyword_id': 2
    }

    movie_link_column = {
        'id': 0,
        'movie_id': 1,
        'linked_movie_id': 2,
        'link_type_id': 3
    }

    name_column = {
        'id': 0,
        'name': 1,
        'imdb_index': 2,
        'imdb_id': 3,
        'gender': 4,
        'name_pcode_cf': 5,
        'name_pcode_nf': 6,
        'surname_pcode': 7,
        'md5sum': 8
    }

    role_type_column = {
        'id': 0,
        'role': 1
    }

    title_column = {
        'id': 0,
        'title': 1,
        'imdb_index': 2,
        'kind_id': 3,
        'production_year': 4,
        'imdb_id': 5,
        'phonetic_code': 6,
        'episode_of_id': 7,
        'season_nr': 8,
        'episode_nr': 9,
        'series_years': 10,
        'md5sum': 11
    }

    movie_info_column = {
        'id': 0,
        'movie_id': 1,
        'info_type_id': 2,
        'info': 3,
        'note': 4
    }

    person_info_column = {
        'id': 0,
        'person_id': 1,
        'info_type_id': 2,
        'info': 3,
        'note': 4
    }
    data["aka_name"].columns = aka_name_column
    data["aka_title"].columns = aka_title_column
    data["cast_info"].columns = cast_info_column
    data["char_name"].columns = char_name_column
    data["company_name"].columns = company_name_column
    data["company_type"].columns = company_type_column
    data["comp_cast_type"].columns = comp_cast_type_column
    data["complete_cast"].columns = complete_cast_column
    data["info_type"].columns = info_type_column
    data["keyword"].columns = keyword_column
    data["kind_type"].columns = kind_type_column
    data["link_type"].columns = link_type_column
    data["movie_companies"].columns = movie_companies_column
    data["movie_info"].columns = movie_info_column
    data["movie_info_idx"].columns = movie_info_idx_column
    data["movie_keyword"].columns = movie_keyword_column
    data["movie_link"].columns = movie_link_column
    data["name"].columns = name_column
    data["person_info"].columns = person_info_column
    data["role_type"].columns = role_type_column
    data["title"].columns = title_column
    return data


def prepare_dataset(database):

    column2pos = dict()

    tables = ['aka_name', 'aka_title', 'cast_info', 'char_name', 'company_name', 'company_type', 'comp_cast_type', 'complete_cast', 'info_type', 'keyword', 'kind_type', 'link_type', 'movie_companies', 'movie_info', 'movie_info_idx',
              'movie_keyword', 'movie_link', 'name', 'person_info', 'role_type', 'title']

    for table_name in tables:
        column2pos[table_name] = database[table_name].columns

    indexes = ['aka_name_pkey', 'aka_title_pkey', 'cast_info_pkey', 'char_name_pkey',
               'comp_cast_type_pkey', 'company_name_pkey', 'company_type_pkey', 'complete_cast_pkey',
               'info_type_pkey', 'keyword_pkey', 'kind_type_pkey', 'link_type_pkey', 'movie_companies_pkey',
               'movie_info_idx_pkey', 'movie_keyword_pkey', 'movie_link_pkey', 'name_pkey', 'role_type_pkey',
               'title_pkey', 'movie_info_pkey', 'person_info_pkey', 'company_id_movie_companies',
               'company_type_id_movie_companies', 'info_type_id_movie_info_idx', 'info_type_id_movie_info',
               'info_type_id_person_info', 'keyword_id_movie_keyword', 'kind_id_aka_title', 'kind_id_title',
               'linked_movie_id_movie_link', 'link_type_id_movie_link', 'movie_id_aka_title', 'movie_id_cast_info',
               'movie_id_complete_cast', 'movie_id_movie_ companies', 'movie_id_movie_info_idx',
               'movie_id_movie_keyword', 'movie_id_movie_link', 'movie_id_movie_info', 'person_id_aka_name',
               'person_id_cast_info', 'person_id_person_info', 'person_role_id_cast_info', 'role_id_cast_info']
    indexes_id = dict()
    for idx, index in enumerate(indexes):
        indexes_id[index] = idx + 1
    physic_ops_id = {'Materialize':1, 'Sort':2, 'Hash':3, 'Merge Join':4, 'Bitmap Index Scan':5,
                     'Index Only Scan':6, 'BitmapAnd':7, 'Nested Loop':8, 'Aggregate':9, 'Result':10,
                     'Hash Join':11, 'Seq Scan':12, 'Bitmap Heap Scan':13, 'Index Scan':14, 'BitmapOr':15}
    strategy_id = {'Plain':1}
    compare_ops_id = {'=':1, '>':2, '<':3, '!=':4, '~~':5, '!~~':6, '!Null': 7, '>=':8, '<=':9}
    bool_ops_id = {'AND':1,'OR':2}
    tables_id = {}
    columns_id = {}
    table_id = 1
    column_id = 1
    for table_name in tables:
        tables_id[table_name] = table_id
        table_id += 1
        for column in column2pos[table_name]:
            columns_id[table_name+'.'+column] = column_id
            column_id += 1
    return column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, tables
