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
        if query_type in ['1-chain', '2-chain', '3-chain', '4-chain', '5-chain']:
            query += "[MASK]" + "".join([" [rela] " + each for each in sample["query_text"][1][::-1]]) + " [rela] " + sample["query_text"][0]
        elif query_type in ["2-inter", "3-inter", "chain-inter", "2-union"]:
            for each in sample["query_text"]:
                query += "[MASK]" + "".join([" [rela] " + _ for _ in each[1][::-1]]) + " [rela] " + each[0] + " " 
        elif query_type in ["inter-chain", "union-chain", "3-inter-chain"]:
            for each in sample["query_text"][:-1]:
                query += "[MASK]" + " [rela] " + sample["query_text"][-1] + "".join([" [rela] " + _ for _ in each[1][::-1]]) +  "[rela] " + each[0] + " " 
        elif query_type in ["inter-2-chain"]:
            for each in sample["query_text"][:-2]:
                query += "[MASK]" + " [rela] " + sample["query_text"][-1] + " [rela] " + sample["query_text"][-2] + "".join([" [rela] " + _ for _ in each[1][::-1]]) +  "[rela] " + each[0] + " " 

        query = query.strip()

        query_inputs = self.tokenizer(query, max_length=self.max_seq_len, 
                            return_tensors="pt",
                            truncation='longest_first')

        input_ids = query_inputs["input_ids"][0].numpy().tolist()
        
        rela_token_id = self.tokenizer.convert_tokens_to_ids("[rela]")
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        rela_count = query.count("[rela]")

        new_input_ids = []
        token_type_ids = []
        attention_mask_ids = []
        positional_ids = []
        # print(self.tokenizer.convert_ids_to_tokens(input_ids))

        mask_position = []
        for i, each_id in enumerate(input_ids):
            if i == 0:
                positional_ids.append(0)
            elif i == len(input_ids) - 1:
                positional_ids.append(510) 
            elif each_id == mask_token_id:
                start = 1
                mask_position.append(len(positional_ids))
                positional_ids.append(start)
            elif each_id == rela_token_id:
                start += 1
            else:
                positional_ids.append(start)

            if each_id != rela_token_id:
                new_input_ids.append(input_ids[i])
                token_type_ids.append(query_inputs["token_type_ids"][0][i])
                attention_mask_ids.append(query_inputs["attention_mask"][0][i])
        
        assert len(new_input_ids) == len(positional_ids)
        assert len(new_input_ids) == len(input_ids) - rela_count
        assert rela_token_id not in new_input_ids

        query_inputs["input_ids"] =  torch.tensor([new_input_ids])
        query_inputs["token_type_ids"] =  torch.tensor([token_type_ids])
        query_inputs["attention_mask"] =  torch.tensor([attention_mask_ids])
        
        return_dict = {
            "query_inputs": query_inputs,
            "position_ids": torch.LongTensor(positional_ids),
            "mask_positions": torch.LongTensor(mask_position)
        }

        if self.train:
            target_index = np.random.choice(range(len(sample["ans_text"])))
            ans_inputs = self.tokenizer("[target] " + sample["ans_text"][target_index], max_length=self.max_ans_len,
                                truncation='longest_first',
                                return_tensors="pt")

            return_dict["ans_inputs"] = ans_inputs

            return_dict["ans"] = sample["ans"]
            return_dict["selected_ans"] = torch.LongTensor([sample["ans"][target_index]])

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
        'query_type_ids': collate_tokens([s["query_inputs"]["token_type_ids"].view(-1) for s in samples], pad_id),
    }

    if "position_ids" in samples[0]:
        batch["position_ids"] = collate_tokens([s["position_ids"] for s in samples], 511) 

    if "ans_inputs" in samples[0]:
        batch.update({
            'ans_input_ids': collate_tokens([s["ans_inputs"]["input_ids"].view(-1) for s in samples], pad_id),
            'ans_masks': collate_tokens([s["ans_inputs"]["attention_mask"].view(-1) for s in samples], pad_id)
        })

    if "selected_ans" in samples[0]:
        negative_index_list = []
        only_negative_index_list = []
        tag_list = []
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
            # hard negative
            # cur_tag += [torch.LongTensor([0]),] * 5 
            # tag_list.append(cur_tag)
        batch["tags"] = collate_tokens(tag_list, -1)

        batch["selected_ans"] = collate_tokens([torch.LongTensor([s["selected_ans"]]) for s in samples], -1)
        batch["ans"] = collate_tokens([torch.LongTensor(s["ans"]) for s in samples], -1)
    
    if "mask_positions" in samples[0]:
        batch["mask_positions"] = collate_tokens([s["mask_positions"] for s in samples], 0) 
    
    if "index" in samples[0]:
        batch["index"] = collate_tokens([s["index"] for s in samples], -1)
    
    return batch


            