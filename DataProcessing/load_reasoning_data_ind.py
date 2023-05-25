import argparse
import logging
import pickle
from tqdm import tqdm
import json
import numpy as np
import random


def get_relation2text(relation2text, rela):
    if rela[0] == '+':
        return relation2text[rela[1:]]
    elif rela[0] == '-':
        return "inverse " + relation2text[rela[1:]]
    else:
        raise Exception('error')

def get_relation2text_v2(rela):
    assert rela[1:9] == "concept:"
    if rela[0] == '+':
        rela_text = rela[9:]
    elif rela[0] == '-':
        rela_text = "inverse " + rela[9:]
    else:
        assert "haha"
    return rela_text

def get_entity2text(entity):
    if entity[:8] == 'concept:':
        entity_split = entity[8:].split(":")
        entity_split_2 = entity_split[1].split("_")
        entity_split_2 = [each.strip() for each in entity_split_2 if len(each.strip()) > 0]
        entity_text = " ".join(entity_split[:1] + entity_split_2)
    else:
        entity_text = entity
    print(entity_text)
    return entity_text


def read_data(params, add_union=False, other_type=False):
    data_path = f'../Data/{params.dataset}'

    with open(f'{data_path}/id2ent.pkl', 'rb') as ent_handle:
        ind2ents = pickle.load(ent_handle)

    with open(f'{data_path}/id2rel.pkl', 'rb') as rel_handle:
        ind2rels = pickle.load(rel_handle)

    text_data_dir = '../Data/text/FB237'
    entity2text = {}
    with open(f'{text_data_dir}/entity2text.txt') as fin:
        for l in fin:
            entity, text = l.strip().split('\t')
            name = text.split(',')[0]
            entity2text[entity] = name
    relation2text = {}
    with open(f'{text_data_dir}/relation2text.txt') as fin:
        for l in fin:
            relation, text = l.strip().split('\t')
            relation2text[relation] = text

    all_triple_list = []
    if params.set_type != 'train':
        task_list = [('e', ('r',)), ('e', ('r', 'r')), ('e', ('r', 'r', 'r')), (('e', ('r',)), ('e', ('r',))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',))), (('e', ('r', 'r')), ('e', ('r',))), ((('e', ('r',)), ('e', ('r',))), ('r',)), (('e', ('r',)), ('e', ('r',)), ('u',)), ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))]
    else:
        task_list = [('e', ('r',)), ('e', ('r', 'r')), ('e', ('r', 'r', 'r')), (('e', ('r',)), ('e', ('r',))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',)))]

    task_type_list = ['1-chain', '2-chain', '3-chain', '2-inter', '3-inter', 'chain-inter', 'inter-chain', '2-union', 'union-chain']
    data_list = []

    if params.set_type=='train':
        with open(f'{data_path}/train-queries.pkl', 'rb') as handle:
            triples_list = pickle.load(handle)
        with open(f'{data_path}/train-answers.pkl', 'rb') as handle:
            ans_dict = pickle.load(handle)
    elif params.set_type=='dev':
        with open(f'{data_path}/valid-queries.pkl', 'rb') as handle:
            triples_list = pickle.load(handle)
        with open(f'{data_path}/valid-easy-answers.pkl', 'rb') as handle:
            ans_dict = pickle.load(handle)
        with open(f'{data_path}/valid-hard-answers.pkl', 'rb') as handle:
            hard_ans_dict = pickle.load(handle)
    else:
        with open(f'{data_path}/test-queries.pkl', 'rb') as handle:
            triples_list = pickle.load(handle)
        with open(f'{data_path}/test-easy-answers.pkl', 'rb') as handle:
            ans_dict = pickle.load(handle)
        with open(f'{data_path}/test-hard-answers.pkl', 'rb') as handle:
            hard_ans_dict = pickle.load(handle)

    type_1c_list = []
    type_2c_list = []
    type_3c_list = []
    type_2i_list = []
    type_3i_list = []
    index = 0
    for n, task in enumerate(task_list): 
        cur_triples_list = triples_list[task]

        for i, each in tqdm(enumerate(cur_triples_list)):
            if n == 0:
                type_1c_list.append(index)
            elif n == 1:
                type_2c_list.append(index)
            elif n == 2:
                type_3c_list.append(index)
            elif n == 3:
                type_2i_list.append(index)
            elif n == 4:
                type_3i_list.append(index)

            cur_inst = {}
            cur_inst['id'] = index
            cur_inst['type'] = task_type_list[n]
            if cur_inst['type'] in ['1-chain', '2-chain', '3-chain']:
                cur_inst['query'] = each
                head, rela = cur_inst['query']
                cur_inst['ans'] = list(ans_dict[each])

                if params.set_type != 'train':
                    cur_inst['hard_ans'] = list(hard_ans_dict[each])

                if "FB15k" in params.dataset:
                    head_text = entity2text[ind2ents[head]]
                    rela_text = tuple([get_relation2text(relation2text, ind2rels[each]) for each in rela])
                    cur_inst['query_text'] = (head_text, rela_text)

                    cur_inst['ans_text'] = [entity2text[ind2ents[each]] for each in cur_inst['ans']]
                    if params.set_type != 'train':
                        cur_inst['hard_ans_text'] = [entity2text[ind2ents[each]] for each in cur_inst['hard_ans']]
                else:
                    head_text = get_entity2text(ind2ents[head])
                    rela_text = tuple([get_relation2text_v2(ind2rels[each]) for each in rela])
                    cur_inst['query_text'] = (head_text, rela_text)

                    cur_inst['ans_text'] = [get_entity2text(ind2ents[each]) for each in cur_inst['ans']]
                    if params.set_type != 'train':
                        cur_inst['hard_ans_text'] = [get_entity2text(ind2ents[each]) for each in cur_inst['hard_ans']]

            else:
                if cur_inst['type'] in ['inter-chain']:
                    cur_inst['query'] = each[0] + each[1]
                elif cur_inst['type'] in ['2-union']:
                    cur_inst['query'] = each[:2]
                elif cur_inst['type'] in ['union-chain']:
                    cur_inst['query'] = each[0][:2] + each[1]
                else:
                    cur_inst['query'] = each
                cur_inst['ans'] = list(ans_dict[each])

                if params.set_type != 'train':
                    cur_inst['hard_ans'] = list(hard_ans_dict[each])

                if "FB15k" in params.dataset:
                    query_text = []
                    for each_item in cur_inst['query']:
                        if type(each_item) == tuple:
                            head, rela = each_item
                            head_text = entity2text[ind2ents[head]]
                            rela_text = tuple([get_relation2text(relation2text, ind2rels[each]) for each in rela])
                            query_text.append((head_text, rela_text))
                        else:
                            query_text.append(get_relation2text(relation2text, ind2rels[each_item]))
                    cur_inst['query_text'] = tuple(query_text)
                
                    cur_inst['ans_text'] = [entity2text[ind2ents[each]] for each in cur_inst['ans']]
                    if params.set_type != 'train':
                        cur_inst['hard_ans_text'] = [entity2text[ind2ents[each]] for each in cur_inst['hard_ans']]
                else:
                    query_text = []
                    for each_item in cur_inst['query']:
                        if type(each_item) == tuple:
                            head, rela = each_item
                            head_text = get_entity2text(ind2ents[head])
                            rela_text = tuple([get_relation2text_v2(ind2rels[each]) for each in rela])
                            query_text.append((head_text, rela_text))
                        else:
                            query_text.append(get_relation2text_v2(ind2rels[each_item]))
                    cur_inst['query_text'] = tuple(query_text)
                
                    cur_inst['ans_text'] = [get_entity2text(ind2ents[each]) for each in cur_inst['ans']]
                    if params.set_type != 'train':
                        cur_inst['hard_ans_text'] = [get_entity2text(ind2ents[each]) for each in cur_inst['hard_ans']]

            data_list.append(cur_inst)
            index += 1

    print(len(data_list))
    expand_data_list = []
    print(len(type_1c_list), len(type_2c_list), len(type_3c_list), len(type_2i_list), len(type_3i_list))

    if other_type:
        # 4c
        random.shuffle(type_1c_list)
        for each in tqdm(type_1c_list):
            cur_ans = data_list[each]['ans']
            for each_j in type_3c_list:
                if data_list[each_j]['query'][0] in cur_ans:
                    cur_inst = {}
                    cur_inst['id'] = len(data_list) + len(expand_data_list)
                    cur_inst['type'] = '4-chain'

                    cur_inst['query'] = (data_list[each]['query'][0], data_list[each]['query'][1] + data_list[each_j]['query'][1])
                    cur_inst['query_text'] = (data_list[each]['query_text'][0], data_list[each]['query_text'][1] + data_list[each_j]['query_text'][1])

                    cur_inst['ans'] = data_list[each_j]['ans'] 
                    cur_inst['ans_text'] = data_list[each_j]['ans_text'] 
                    cur_inst['hard_ans'] = data_list[each_j]['hard_ans'] 
                    cur_inst['hard_ans_text'] = data_list[each_j]['hard_ans_text']
                    assert len(cur_inst['hard_ans']) > 0
                    expand_data_list.append(cur_inst)
                    break

        print(len(expand_data_list))

        # 5c
        random.shuffle(type_2c_list)
        for each in tqdm(type_2c_list):
            cur_ans = data_list[each]['ans']
            for each_j in type_3c_list:
                if data_list[each_j]['query'][0] in cur_ans:
                    cur_inst = {}
                    cur_inst['id'] = len(data_list) + len(expand_data_list)
                    cur_inst['type'] = '5-chain'

                    cur_inst['query'] = (data_list[each]['query'][0], data_list[each]['query'][1] + data_list[each_j]['query'][1])
                    cur_inst['query_text'] = (data_list[each]['query_text'][0], data_list[each]['query_text'][1] + data_list[each_j]['query_text'][1])

                    cur_inst['ans'] = data_list[each_j]['ans'] 
                    cur_inst['ans_text'] = data_list[each_j]['ans_text'] 
                    cur_inst['hard_ans'] = data_list[each_j]['hard_ans'] 
                    cur_inst['hard_ans_text'] = data_list[each_j]['hard_ans_text']
                    assert len(cur_inst['hard_ans']) > 0
                    expand_data_list.append(cur_inst)
                    break
        
        print(len(expand_data_list))

        # 3ip
        random.shuffle(type_3i_list)
        for each in tqdm(type_3i_list):
            cur_ans = data_list[each]['ans']
            for each_j in type_1c_list:
                if data_list[each_j]['query'][0] in cur_ans:
                    cur_inst = {}
                    cur_inst['id'] = len(data_list) + len(expand_data_list)
                    cur_inst['type'] = '3-inter-chain'

                    cur_inst['query'] = data_list[each]['query'] + data_list[each_j]['query'][1]
                    cur_inst['query_text'] = data_list[each]['query_text'] + data_list[each_j]['query_text'][1]

                    cur_inst['ans'] = data_list[each_j]['ans'] 
                    cur_inst['ans_text'] = data_list[each_j]['ans_text'] 
                    cur_inst['hard_ans'] = data_list[each_j]['hard_ans'] 
                    cur_inst['hard_ans_text'] = data_list[each_j]['hard_ans_text']
                    assert len(cur_inst['hard_ans']) > 0
                    expand_data_list.append(cur_inst)
                    break
        
        print(len(expand_data_list))

        # i2p
        random.shuffle(type_2i_list)
        for each in tqdm(type_2i_list):
            cur_ans = data_list[each]['ans']
            for each_j in type_2c_list:
                if data_list[each_j]['query'][0] in cur_ans:
                    cur_inst = {}
                    cur_inst['id'] = len(data_list) + len(expand_data_list)
                    cur_inst['type'] = 'inter-2-chain'

                    cur_inst['query'] = data_list[each]['query'] + data_list[each_j]['query'][1]
                    cur_inst['query_text'] = data_list[each]['query_text'] + data_list[each_j]['query_text'][1]

                    cur_inst['ans'] = data_list[each_j]['ans'] 
                    cur_inst['ans_text'] = data_list[each_j]['ans_text'] 
                    cur_inst['hard_ans'] = data_list[each_j]['hard_ans'] 
                    cur_inst['hard_ans_text'] = data_list[each_j]['hard_ans_text']
                    assert len(cur_inst['hard_ans']) > 0
                    expand_data_list.append(cur_inst)
                    break

        print(len(expand_data_list))

    if add_union:
        # 2u 
        for i, each in tqdm(enumerate(type_1c_list)):
            union_index = np.random.choice(type_1c_list[:i] + type_1c_list[i+1:]) 

            cur_inst = {}
            cur_inst['id'] = len(data_list) + len(expand_data_list)
            cur_inst['type'] = "2-union"
            cur_inst['query'] = (data_list[each]['query'], data_list[union_index]['query'])
            cur_inst['query_text'] = (data_list[each]['query_text'], data_list[union_index]['query_text'])
            cur_inst['ans'] = data_list[each]['ans'] + data_list[union_index]['ans']
            cur_inst['first_ans'] = data_list[each]['ans']
            cur_inst['second_ans'] = data_list[union_index]['ans']
            cur_inst['ans_text'] = data_list[each]['ans_text'] + data_list[union_index]['ans_text']

            expand_data_list.append(cur_inst)

    
    data_list = data_list + expand_data_list
    print(len(data_list), len(expand_data_list))
    if add_union:
        dump_data = {'data': data_list}
        with open(f'{data_path}/{params.set_type}_addunion.json', 'w') as w_f:
            json.dump(dump_data, w_f, indent=2)
    else:
        dump_data = {'data': data_list}
        with open(f'{data_path}/{params.set_type}.json', 'w') as w_f:
            json.dump(dump_data, w_f, indent=2)
        
        if other_type:
            dump_data = {'data': expand_data_list}
            with open(f'{data_path}/{params.set_type}_othertypes.json', 'w') as w_f:
                json.dump(dump_data, w_f, indent=2)


def get_all_entity_texts(params):
    output = {}

    data_path = f'../Data/{params.dataset}/'
    with open(f'{data_path}/stats.txt') as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
    
    with open(f'{data_path}/id2ent.pkl', 'rb') as ent_handle:
        ind2ents = pickle.load(ent_handle)

    text_data_dir = '../Data/text/FB237'
    entity2text = {}
    with open(f'{text_data_dir}/entity2text.txt') as fin:
        for l in fin:
            entity, text = l.strip().split('\t')
            name = text.split(',')[0]
            entity2text[entity] = name
    for i in range(nentity):
        entity_text = entity2text[ind2ents[i]]
        output[i] = entity_text
    
    with open(f'{data_path}/entity_text.json', 'w') as w_f:
        json.dump(output, w_f, indent=2)


def split_dev_dataset(params, set_type, other_type=False):
    data_path = f'../Data/{params.dataset}'
    if other_type:
        with open(f'{data_path}/{set_type}_othertypes.json', "r") as r_f:
            train_data = json.load(r_f)['data']
    else:
        with open(f'{data_path}/{set_type}.json', "r") as r_f:
            train_data = json.load(r_f)['data']

    data_list_1c = []
    data_list_2c = []
    data_list_3c = []
    data_list_2i = []
    data_list_3i = []
    data_list_ip = []
    data_list_pi = []
    data_list_2u = []
    data_list_up = []

    data_list_4c = []
    data_list_5c = []
    data_list_3ip = []
    data_list_i2p = []

    for each in train_data:
        if each['type'] == '1-chain':
            data_list_1c.append(each)
        elif each['type'] == '2-chain':
            data_list_2c.append(each)
        elif each['type'] == '3-chain':
            data_list_3c.append(each)
        elif each['type'] == '2-inter':
            data_list_2i.append(each)
        elif each['type'] == '3-inter':
            data_list_3i.append(each)
        elif each['type'] == 'inter-chain':
            data_list_ip.append(each)
        elif each['type'] == 'chain-inter':
            assert len(each["query"]) == 2
            assert len(each["query"][0][1]) == 2 and len(each["query"][1][1]) == 1
            data_list_pi.append(each)
        elif each['type'] == '2-union':
            data_list_2u.append(each)
        elif each['type'] == 'union-chain':
            data_list_up.append(each)
        elif each['type'] == '4-chain':
            data_list_4c.append(each)
        elif each['type'] == '5-chain':
            data_list_5c.append(each)
        elif each['type'] == '3-inter-chain':
            data_list_3ip.append(each)
        else:
            data_list_i2p.append(each)
    
    print(len(data_list_1c), len(data_list_2c), len(data_list_3c), len(data_list_2i), len(data_list_3i))
    print(len(data_list_ip), len(data_list_pi), len(data_list_2u), len(data_list_up))
    print(len(data_list_4c), len(data_list_5c), len(data_list_3ip), len(data_list_i2p))

    if other_type:
        with open(f'{data_path}/{set_type}_4c.json', 'w') as w_f_10:
            json.dump({'data': data_list_4c}, w_f_10, indent=2)

        with open(f'{data_path}/{set_type}_5c.json', 'w') as w_f_11:
            json.dump({'data': data_list_5c}, w_f_11, indent=2)

        with open(f'{data_path}/{set_type}_3ip.json', 'w') as w_f_12:
            json.dump({'data': data_list_3ip}, w_f_12, indent=2)

        with open(f'{data_path}/{set_type}_i2p.json', 'w') as w_f_13:
            json.dump({'data': data_list_i2p}, w_f_13, indent=2)
    else:
        with open(f'{data_path}/{set_type}_1c.json', 'w') as w_f_1:
            json.dump({'data': data_list_1c}, w_f_1, indent=2)

        with open(f'{data_path}/{set_type}_2c.json', 'w') as w_f_2:
            json.dump({'data': data_list_2c}, w_f_2, indent=2)

        with open(f'{data_path}/{set_type}_3c.json', 'w') as w_f_3:
            json.dump({'data': data_list_3c}, w_f_3, indent=2)

        with open(f'{data_path}/{set_type}_2i.json', 'w') as w_f_4:
            json.dump({'data': data_list_2i}, w_f_4, indent=2)

        with open(f'{data_path}/{set_type}_3i.json', 'w') as w_f_5:
            json.dump({'data': data_list_3i}, w_f_5, indent=2)
        
        with open(f'{data_path}/{set_type}_ip.json', 'w') as w_f_6:
            json.dump({'data': data_list_ip}, w_f_6, indent=2)

        with open(f'{data_path}/{set_type}_pi.json', 'w') as w_f_7:
            json.dump({'data': data_list_pi}, w_f_7, indent=2)

        with open(f'{data_path}/{set_type}_2u.json', 'w') as w_f_8:
            json.dump({'data': data_list_2u}, w_f_8, indent=2)

        with open(f'{data_path}/{set_type}_up.json', 'w') as w_f_9:
            json.dump({'data': data_list_up}, w_f_9, indent=2)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='BERTRL model')
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")

    parser.add_argument('--set_type', '-st', type=str, default='train',
                        help='set type of train/valid/test')

    params = parser.parse_args()

    params.dataset = "Ind-FB15k-237-V2"
    params.set_type = "train"
    read_data(params, add_union=True)
    
    params.set_type = "dev"
    read_data(params)

    params.set_type = "test"
    read_data(params, other_type=True)

    get_all_entity_texts(params)
    split_dev_dataset(params, "dev")
    split_dev_dataset(params, "test")
    split_dev_dataset(params, "test", other_type=True)
    

    params.dataset = "Ind-NELL-V3"
    params.set_type = "train"
    read_data(params, add_union=True)
    
    params.set_type = "dev"
    read_data(params)

    params.set_type = "test"
    read_data(params, other_type=True)

    get_all_entity_texts(params)
    split_dev_dataset(params, "dev")
    split_dev_dataset(params, "test")
    split_dev_dataset(params, "test", other_type=True)