from torch.utils.data import Dataset
import json
import torch
import numpy as np


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    # print(len(values), size)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class ReasoningDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 data_path,
                 max_seq_len,
                 max_ans_len,
                 negative_num=5,
                 train=False,
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_ans_len = max_ans_len
        self.negative_num = negative_num
        self.train = train
        print(f"Loading data from {data_path}")

        with open(data_path, "r") as r_f:
            self.data = json.load(r_f)['data']

        # Structural prompt
        self.type_descriptions = {
            '1-chain': "one step: [projection]", 
            '2-chain': "two steps: [projection], then [projection]", 
            '3-chain': "three steps: [projection], then [projection], then [projection]", 
            '2-inter': "three steps: [projection], and [projection], [intersection]", 
            '3-inter': "four steps: [projection], and [projection], and [projection], [intersection]",
            'chain-inter': "four steps: [projection], then [projection], and [projection], [intersection]",
            'inter-chain': "five steps: [projection], then [projection], and [projection], then [projection], [intersection]",
            '2-union': "three steps: [projection], and [projection], [union]", 
            'union-chain': "five steps: [projection], then [projection], and [projection], then [projection], [union]"
        }

        entity_texts_file = "/".join(data_path.split("/")[:-1]) + "/entity_text.json"
        print(f"Loading entity texts from {entity_texts_file}")
        with open(entity_texts_file, "r") as e_r_f:
            self.entities = json.load(e_r_f)

        if not train:
            self.features = {}
            candidate_input_ids_list = []
            candidate_atten_masks_list = []
            for i in range(len(self.entities)):   
                candidate_inputs = self.tokenizer("[target] " + self.entities[str(i)], max_length=self.max_ans_len,
                            truncation='longest_first',
                            return_tensors="pt")
                candidate_input_ids_list.append(candidate_inputs["input_ids"])
                candidate_atten_masks_list.append(candidate_inputs["attention_mask"])
            
            self.features['entity_input_ids'] = collate_tokens([s.view(-1) for s in candidate_input_ids_list], self.tokenizer.pad_token_id)
            self.features['entity_masks'] = collate_tokens([s.view(-1) for s in candidate_atten_masks_list], self.tokenizer.pad_token_id)


    def __getitem__(self, index):
        sample = self.data[index]
        query_type = sample["type"]

        query = ""
        if query_type in ['1-chain', '2-chain', '3-chain']:
            query += "[projection] [anchor] " + sample["query_text"][0] + "".join([" [projection] [rela] " + each if i > 0 else " [rela] " + each for i, each in enumerate(sample["query_text"][1])])
        elif query_type in ["2-inter", "3-inter", "chain-inter", "2-union"]:
            for each in sample["query_text"]:
                query += "[projection] [anchor] " + each[0] + "".join([" [projection] [rela] " + _ if i > 0 else " [rela] " + _ for i, _ in enumerate(each[1])]) + " "
            if query_type in ["2-inter", "3-inter", "chain-inter"]:
                query = "[intersection] " + query
            else:
                query = "[union] " + query
        elif query_type in ["inter-chain", "union-chain"]:
            for each in sample["query_text"][:-1]:
                query += "[projection] [anchor] " + each[0] + "".join([" [projection] [rela] " + _ if i > 0 else " [rela] " + _ for i, _ in enumerate(each[1])]) + " [projection] [rela] " + sample["query_text"][-1] + " " #" [projection] " 
                if query_type == "inter-chain":
                    query = "[intersection] " + query 
                else:
                    query = "[union] " + query 

        query = "[qtype] " + self.type_descriptions[query_type] + " [SEP] " + query

        query = query.strip()
        query_inputs = self.tokenizer(query, max_length=self.max_seq_len, 
                            return_tensors="pt",
                            truncation='longest_first')

        input_ids = query_inputs["input_ids"][0].numpy().tolist()

        anchor_token_id = self.tokenizer.convert_tokens_to_ids("[anchor]")
        rela_token_id = self.tokenizer.convert_tokens_to_ids("[rela]")
        
        entity_positions = []
        relation_positions = []

        edit_entity_positions = []
        edit_relation_positions = []
        if query_type in ['1-chain', '2-chain', '3-chain']:
            entity_positions.append(input_ids.index(anchor_token_id))
            rela_count = query.count("[rela]")
            find_start = 0
            for i in range(rela_count):
                cur_rela_loc = input_ids.index(rela_token_id, find_start)
                find_start = cur_rela_loc + 1
                if i == 0:
                    entity_positions.append(cur_rela_loc - 1)
                else:
                    relation_positions.append(cur_rela_loc - 2)
                
                relation_positions.append(cur_rela_loc)
            relation_positions.append(len(input_ids)-2)
            
            if query_type == '1-chain':
                edit_entity_positions = entity_positions[:1] * 12
                edit_relation_positions = relation_positions[1:2] * 12
            elif query_type == '2-chain':
                edit_entity_positions = entity_positions[:1] * 12
                edit_relation_positions = relation_positions[3:4] * 12
            elif query_type == '3-chain':
                edit_entity_positions = entity_positions[:1] * 12
                edit_relation_positions = relation_positions[5:6] * 12

        elif query_type in ["2-inter", "3-inter", "chain-inter", "2-union"]:
            anchor_count = query.count("[anchor]")
            rela_count = query.count("[rela]")
            find_anchor_start = 0
            find_rela_start = 0

            if sample["type"] == "chain-inter":
                assert anchor_count == 2
                assert rela_count == 3
                for i in range(anchor_count):
                    cur_anchor_loc = input_ids.index(anchor_token_id, find_anchor_start)
                    cur_rela_loc = input_ids.index(rela_token_id, find_rela_start)
                    find_anchor_start = cur_anchor_loc + 1
                    find_rela_start = cur_rela_loc + 1

                    entity_positions.append(cur_anchor_loc)
                    entity_positions.append(cur_rela_loc - 1)

                    if i > 0:
                        relation_positions.append(cur_anchor_loc - 2)
                    relation_positions.append(cur_rela_loc)

                    if i == 0:
                        cur_rela_loc_2 = input_ids.index(rela_token_id, find_rela_start)
                        find_rela_start = cur_rela_loc_2 + 1
                        relation_positions.append(cur_rela_loc_2 - 2)
                        relation_positions.append(cur_rela_loc_2)
                relation_positions.append(len(input_ids)-2)
                assert len(entity_positions) == 4 
                assert len(relation_positions) == 6
            else:
                inter_num = int(sample["type"].split('-')[0])

                assert anchor_count == inter_num
                assert rela_count == inter_num 
                for i in range(anchor_count):
                    cur_anchor_loc = input_ids.index(anchor_token_id, find_anchor_start)
                    cur_rela_loc = input_ids.index(rela_token_id, find_rela_start)
                    find_anchor_start = cur_anchor_loc + 1
                    find_rela_start = cur_rela_loc + 1

                    entity_positions.append(cur_anchor_loc)
                    entity_positions.append(cur_rela_loc - 1)
                    
                    if i > 0:
                        relation_positions.append(cur_anchor_loc - 2)
                    relation_positions.append(cur_rela_loc)
                relation_positions.append(len(input_ids)-2)
            
            if query_type == '2-inter':
                edit_entity_positions = entity_positions[:1] + entity_positions[2:3]
                edit_relation_positions = relation_positions[1:2] + relation_positions[3:4] 
                edit_entity_positions = edit_entity_positions * 6
                edit_relation_positions = edit_relation_positions * 6
            elif query_type == '3-inter':
                edit_entity_positions = entity_positions[:1] + entity_positions[2:3] + entity_positions[4:5]
                edit_relation_positions = relation_positions[1:2] + relation_positions[3:4] + relation_positions[5:6]
                edit_entity_positions = edit_entity_positions * 4
                edit_relation_positions = edit_relation_positions * 4
            elif query_type == 'chain-inter':
                edit_entity_positions = entity_positions[:1] + entity_positions[2:3] 
                edit_relation_positions = relation_positions[3:4] + relation_positions[5:6] 
                edit_entity_positions = edit_entity_positions * 6
                edit_relation_positions = edit_relation_positions * 6
            elif query_type == '2-union':
                edit_entity_positions = entity_positions[:1] * 6 + entity_positions[2:3] * 6
                edit_relation_positions = relation_positions[1:2] * 6 + relation_positions[3:4] * 6
        else:
            anchor_count = query.count("[anchor]")
            rela_count = query.count("[rela]")
            find_anchor_start = 0
            find_rela_start = 0

            assert anchor_count == 2
            assert rela_count == 4

            for i in range(anchor_count):
                cur_anchor_loc = input_ids.index(anchor_token_id, find_anchor_start)
                cur_rela_loc = input_ids.index(rela_token_id, find_rela_start)
                find_anchor_start = cur_anchor_loc + 1
                find_rela_start = cur_rela_loc + 1

                entity_positions.append(cur_anchor_loc)
                entity_positions.append(cur_rela_loc - 1)

                if i > 0:
                    relation_positions.append(cur_anchor_loc - 2)
                relation_positions.append(cur_rela_loc)

                cur_rela_loc_2 = input_ids.index(rela_token_id, find_rela_start)
                find_rela_start = cur_rela_loc_2 + 1

                relation_positions.append(cur_rela_loc_2 - 2)
                relation_positions.append(cur_rela_loc_2)

            relation_positions.append(len(input_ids)-2)
            
            assert len(entity_positions) == 4 
            assert len(relation_positions) == 8

            if query_type == 'inter-chain':
                edit_entity_positions = entity_positions[:1] + entity_positions[2:3] 
                edit_relation_positions = relation_positions[3:4] + relation_positions[7:8] 
                edit_entity_positions = edit_entity_positions * 6
                edit_relation_positions = edit_relation_positions * 6
            else:
                edit_entity_positions = entity_positions[:1] * 6 + entity_positions[2:3] * 6
                edit_relation_positions = relation_positions[3:4] * 6 + relation_positions[7:8] * 6

        type_index = list(self.type_descriptions.keys()).index(query_type)
        return_dict = {
            "query_inputs": query_inputs,
            "entity_positions": torch.LongTensor(edit_entity_positions),
            "relation_positions": torch.LongTensor(edit_relation_positions),
            'type': torch.LongTensor([type_index])
        }


        if self.train:        
            target_index = np.random.choice(range(len(sample["ans_text"])))
            ans_inputs = self.tokenizer("[target] " + sample["ans_text"][target_index], max_length=self.max_ans_len,
                                truncation='longest_first',
                                return_tensors="pt")

            return_dict["ans_inputs"] = ans_inputs

            return_dict["ans"] = sample["ans"]
            return_dict["selected_ans"] = sample["ans"][target_index]

            if query_type == "2-union":
                if sample["ans"][target_index] in sample["first_ans"]:
                    return_dict["union_label"] = torch.tensor([1, 0]) 
                else:
                    return_dict["union_label"] = torch.tensor([0, 1]) 

        if not self.train:
            return_dict["index"] = torch.LongTensor([index])

        return return_dict

    def __len__(self):
        return len(self.data)


def qa_collate(samples, pad_id=0, negative_num=20):
    if len(samples) == 0:
        return {}

    batch = {
        'query_input_ids': collate_tokens([s["query_inputs"]["input_ids"].view(-1) for s in samples], pad_id),
        'query_mask': collate_tokens([s["query_inputs"]["attention_mask"].view(-1) for s in samples], pad_id),
    }

    if "ans_inputs" in samples[0]:
        batch.update({
            'ans_input_ids': collate_tokens([s["ans_inputs"]["input_ids"].view(-1) for s in samples], pad_id),
            'ans_masks': collate_tokens([s["ans_inputs"]["attention_mask"].view(-1) for s in samples], pad_id)
        })

    if "neg_input_ids" in samples[0]:
        neg_ans_input_ids_list = []
        neg_ans_masks_list = []
        for i in range(len(samples[0]["neg_input_ids"])):
            neg_ans_input_ids_list += [s["neg_input_ids"][i].view(-1) for s in samples]
            neg_ans_masks_list += [s["neg_atten_masks"][i].view(-1) for s in samples]
        batch["neg_ans_input_ids"] = collate_tokens(neg_ans_input_ids_list, pad_id)
        batch["neg_ans_masks"] = collate_tokens(neg_ans_masks_list, pad_id)

    if "selected_ans" in samples[0]:
        negative_index_list = []
        only_negative_index_list = []
        for i in range(len(samples)):
            # pos random
            if len(samples) > negative_num:
                cur_nega_index = list(np.random.choice(list(set(range(len(samples)))-{i}), negative_num, replace=False))
            else:
                cur_nega_index = list(np.random.choice(list(set(range(len(samples)))-{i}), negative_num, replace=True))
            only_negative_index_list.append(torch.LongTensor(cur_nega_index))

            insert_index = np.random.choice(range(negative_num+1))
            new_cur_nega_index = cur_nega_index[:insert_index] + [i] + cur_nega_index[insert_index:]
            negative_index_list.append(torch.LongTensor(new_cur_nega_index))
            
        batch["negative_index"] = collate_tokens(negative_index_list, -1)
        batch["true_negative_index"] = collate_tokens(only_negative_index_list, -1)

        tag_list = []
        for i in range(len(samples)):
            # pos random
            cur_tag = []
            pos_num= 0
            for j in range(negative_num+1):
                if samples[negative_index_list[i][j]]["selected_ans"] in samples[i]["ans"]:
                    cur_tag.append(1.0)
                    pos_num += 1
                else:
                    cur_tag.append(0)
            tag_list.append(torch.tensor(cur_tag)/pos_num)
        batch["tags"] = collate_tokens(tag_list, -1)

        batch["selected_ans"] = collate_tokens([torch.LongTensor([s["selected_ans"]]) for s in samples], -1)
        batch["ans"] = collate_tokens([torch.LongTensor(s["ans"]) for s in samples], -1)


    if "index" in samples[0]:
        batch["index"] = collate_tokens([s["index"] for s in samples], -1)

    if "sep_index" in samples[0]:
        batch["sep_index"] = collate_tokens([s["sep_index"] for s in samples], -1)
    
    if "subsampling_weight" in samples[0]:
        batch["subsampling_weight"] = collate_tokens([s["subsampling_weight"] for s in samples], -1)

    if "entity_positions" in samples[0]:
        batch["entity_positions"] = collate_tokens([s["entity_positions"] for s in samples], 0)
        batch["relation_positions"] = collate_tokens([s["relation_positions"] for s in samples], 0)
    
    if "mask_positions" in samples[0]:
        batch["mask_positions"] = collate_tokens([s["mask_positions"] for s in samples], 0)

    if "type" in samples[0]:
        batch["type"] = collate_tokens([s["type"] for s in samples], -1)

    if "union_label" in samples[0]:
        batch["union_label"] = collate_tokens([s["union_label"] for s in samples], -1)
    
    return batch