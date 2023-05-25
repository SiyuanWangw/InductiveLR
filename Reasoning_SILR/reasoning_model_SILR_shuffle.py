import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel
from transformers.trainer_pt_utils import nested_numpify, nested_concat, distributed_concat


def _pad_across_processes(tensor, rank, pad_index=-100):
    """
    Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
    they can safely be gathered.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(_pad_across_processes(t, rank, pad_index=pad_index) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: _pad_across_processes(v, rank, pad_index=pad_index) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )

    if len(tensor.shape) < 2:
        return tensor

    # Gather all sizes
    size = torch.tensor(tensor.shape, device=tensor.device)[None]
    sizes = _nested_gather(size, rank).cpu()

    max_size = max(s[1] for s in sizes)
    if tensor.shape[1] == max_size:
        return tensor

    # Then pad to the maximum size
    old_size = tensor.shape
    new_size = list(old_size)
    new_size[1] = max_size
    new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
    new_tensor[:, : old_size[1]] = tensor
    return new_tensor


def _nested_gather(tensors, rank):
    """
    Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
    concatenating them to `gathered`
    """
    if tensors is None:
        return
    if rank != -1:
        tensors = distributed_concat(tensors)
    return tensors


def gather_data(feature, rank):
    gather_feature_host = None
    gather_feature = _pad_across_processes(feature, rank)
    gather_feature = _nested_gather(gather_feature, rank)
    gather_feature_host = gather_feature if gather_feature_host is None else nested_concat(gather_feature_host,
                                                                gather_feature,
                                                                padding_index=-100)
    return gather_feature_host


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class ReasoningModel(BertPreTrainedModel):
    @staticmethod
    def euclidean_distance(rep1, rep2):
        distance = rep1 - rep2
        distance = torch.norm(distance, p=2, dim=-1)
        return distance

    @staticmethod
    def cross_entropy(p,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        log_likelihood = - torch.log(p[range(m),y]+1e-5)
        loss = torch.sum(log_likelihood) / m
        return loss

    def __init__(self, config, args=None):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(args.model_name)

        self.center_net = CenterIntersection(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size, args.nentity)

    def encode_seq(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        output = self.bert(input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids).last_hidden_state
        pooled_output = torch.mean(output, dim=1)
        return pooled_output, output

    def classifier(self, feature):
        feature = self.dropout(feature)
        logits = self.proj(feature)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, batch, is_prepare=False, lamd=0.3, candidate_rep=None, schema='matching'):
        if is_prepare:
            candidates_rep = self.encode_seq(batch['entity_input_ids'], batch['entity_masks'], batch.get('entity_type_ids', None))[0]
            return candidates_rep
        else:
            query_rep, all_query_rep = self.encode_seq(batch['query_input_ids'], batch['query_mask'], batch.get('query_type_ids', None))
            
            if self.training:
                query_rep_q2b = [[], [], [], [], [], [], [], [], [], [], [], []]
                for i in range(all_query_rep.size(0)):
                    if batch["type"][i][0] in [0,1,2]:
                        query_rep_q2b[0].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][0]+1], dim=0, keepdim=True))
                        for _ in range(1, 12):
                            query_rep_q2b[_].append(query_rep_q2b[0][-1])
                    elif batch["type"][i][0] in [3,5,6]:
                        query_rep_q2b[0].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][0]+1], dim=0, keepdim=True))
                        query_rep_q2b[1].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][1]-1:batch["relation_positions"][i][1]+1], dim=0, keepdim=True))
                        for _ in range(2, 12):
                            query_rep_q2b[_].append(query_rep_q2b[int(_%2)][-1])
                    elif batch["type"][i][0] in [4]:
                        query_rep_q2b[0].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][0]+1], dim=0, keepdim=True))
                        query_rep_q2b[1].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][1]-1:batch["relation_positions"][i][1]+1], dim=0, keepdim=True))                    
                        query_rep_q2b[2].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][2]+1], dim=0, keepdim=True))
                        for _ in range(3, 12):
                            query_rep_q2b[_].append(query_rep_q2b[int(_%3)][-1])
                    elif batch["type"][i][0] in [7,8]:
                        query_rep_q2b[0].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][0]+1], dim=0, keepdim=True))
                        for _ in range(1, 6):
                            query_rep_q2b[_].append(query_rep_q2b[0][-1])
                        query_rep_q2b[6].append(torch.mean(all_query_rep[i][batch["entity_positions"][i][6]-1:batch["relation_positions"][i][6]+1], dim=0, keepdim=True))
                        for _ in range(7, 12):
                            query_rep_q2b[_].append(query_rep_q2b[6][-1])


                for j in range(len(query_rep_q2b)):
                    query_rep_q2b[j] = torch.cat(query_rep_q2b[j], 0)
                query_rep_q2b_1 = self.center_net(torch.stack(query_rep_q2b[:6], dim=0))
                query_rep_q2b_2 = self.center_net(torch.stack(query_rep_q2b[6:], dim=0))

                
                ans_rep = self.encode_seq(batch['ans_input_ids'], batch['ans_masks'], batch.get('a_type_ids', None))[0]
                neg_ans_rep = ans_rep[batch["negative_index"]] 
                neg_ans_rep = neg_ans_rep.transpose(-2, -1)
                tags = batch['tags']

                # cls_query_rep = query_rep.unsqueeze(1)
                # logits = cls_query_rep @ neg_ans_rep
                # logits = logits.squeeze(1)
                # cls_loss_fn = nn.CrossEntropyLoss()
                # rk_loss = cls_loss_fn(logits/1.0, tags)

                cls_loss_fn_2 = nn.CrossEntropyLoss(reduction='none')
                logits_rk_1 = query_rep_q2b_1.unsqueeze(1) @ neg_ans_rep
                logits_rk_1 = logits_rk_1.squeeze(1)
                logits_rk_2 = query_rep_q2b_2.unsqueeze(1) @ neg_ans_rep
                logits_rk_2 = logits_rk_2.squeeze(1)

                rk_loss_1 = cls_loss_fn_2(logits_rk_1/1.0, tags)
                rk_loss_2 = cls_loss_fn_2(logits_rk_2/1.0, tags)

                stack_losses = torch.stack([rk_loss_1, rk_loss_2], dim=1)
                rk_loss = stack_losses * batch["union_label"]
                rk_loss = torch.mean(torch.sum(rk_loss, dim=-1))

                logits1 = self.classifier(query_rep_q2b_1)
                classification_loss_1 = cls_loss_fn_2(logits1, batch['selected_ans'].squeeze())
                logits2 = self.classifier(query_rep_q2b_2)
                classification_loss_2 = cls_loss_fn_2(logits2, batch['selected_ans'].squeeze())
                stack_losses = torch.stack([classification_loss_1, classification_loss_2], dim=1)
                classification_loss = stack_losses * batch["union_label"]
                classification_loss = torch.mean(torch.sum(classification_loss, dim=-1))
            
                return lamd * classification_loss + 1 * rk_loss
            else:
                if batch["type"][0][0] == 0:
                    query_rep_q2b = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][1]+1], dim=0, keepdim=True))
                    query_rep_q2b = torch.cat(query_rep_q2b, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b, query_rep_q2b], dim=0))
                elif batch["type"][0][0] == 1:
                    query_rep_q2b = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                    query_rep_q2b = torch.cat(query_rep_q2b, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b, query_rep_q2b], dim=0))
                elif batch["type"][0][0] == 2:
                    query_rep_q2b = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][5]+1], dim=0, keepdim=True))
                    query_rep_q2b = torch.cat(query_rep_q2b, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b, query_rep_q2b], dim=0))
                elif batch["type"][0][0] == 3:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][1]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b_1, query_rep_q2b_2], dim=0))
                elif batch["type"][0][0] == 4:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    query_rep_q2b_3 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][1]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                        query_rep_q2b_3.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][4]-1:batch["relation_positions"][i][5]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b_3 = torch.cat(query_rep_q2b_3, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b_1, query_rep_q2b_2, query_rep_q2b_3], dim=0))
                elif batch["type"][0][0] == 5:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][5]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b_1, query_rep_q2b_2], dim=0))
                elif batch["type"][0][0] == 6:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][7]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b_1, query_rep_q2b_2], dim=0))
                elif batch["type"][0][0] == 7:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][1]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b = torch.stack([query_rep_q2b_1, query_rep_q2b_2], dim=0)
                elif batch["type"][0][0] == 8:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][7]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b = torch.stack([query_rep_q2b_1, query_rep_q2b_2], dim=0)
                elif batch["type"][0][0] == 9:
                    query_rep_q2b = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][7]+1], dim=0, keepdim=True))
                    query_rep_q2b = torch.cat(query_rep_q2b, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b, query_rep_q2b], dim=0))
                elif batch["type"][0][0] == 10:
                    query_rep_q2b = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][9]+1], dim=0, keepdim=True))
                    query_rep_q2b = torch.cat(query_rep_q2b, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b, query_rep_q2b], dim=0))
                elif batch["type"][0][0] == 11:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    query_rep_q2b_3 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][3]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][7]+1], dim=0, keepdim=True))
                        query_rep_q2b_3.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][4]-1:batch["relation_positions"][i][11]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b_3 = torch.cat(query_rep_q2b_3, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b_1, query_rep_q2b_2, query_rep_q2b_3], dim=0))
                elif batch["type"][0][0] == 12:
                    query_rep_q2b_1 = list()
                    query_rep_q2b_2 = list()
                    for i in range(all_query_rep.size(0)):
                        query_rep_q2b_1.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][0]-1:batch["relation_positions"][i][5]+1], dim=0, keepdim=True))
                        query_rep_q2b_2.append(torch.mean(all_query_rep[i][batch["entity_positions"][i][2]-1:batch["relation_positions"][i][11]+1], dim=0, keepdim=True))
                    query_rep_q2b_1 = torch.cat(query_rep_q2b_1, 0)
                    query_rep_q2b_2 = torch.cat(query_rep_q2b_2, 0)
                    query_rep_q2b = self.center_net(torch.stack([query_rep_q2b_1, query_rep_q2b_2], dim=0))

                # eval via matching for inductive
                if schema=='matching':
                    neg_ans_rep = candidate_rep.unsqueeze(0).repeat(query_rep.size(0), 1, 1).transpose(-2, -1)
                    if batch["type"][0][0] < 7 or batch["type"][0][0] > 8:
                        query_rep_q2b = query_rep_q2b.unsqueeze(1)
                        logits = (query_rep_q2b @ neg_ans_rep).squeeze(1) / 1.0
                        logits = logits.softmax(dim=-1)
                        return logits
                    else:
                        query_rep_1 = query_rep_q2b[0].unsqueeze(1)
                        logits_1 = (query_rep_1 @ neg_ans_rep).squeeze(1) / 1.0
                        score_1 = logits_1.softmax(dim=-1)
                        
                        query_rep_2 = query_rep_q2b[1].unsqueeze(1)
                        logits_2 = (query_rep_2 @ neg_ans_rep).squeeze(1) / 1.0
                        score_2 = logits_2.softmax(dim=-1)
                        
                        score = torch.max(torch.stack([score_1, score_2], dim=0), dim=0)[0]
                        return score
                else:
                    # eval via classification for transductive
                    if batch["type"][0][0] < 7 or batch["type"][0][0] > 8:
                        logits = self.classifier(query_rep_q2b)
                        logits = logits.softmax(dim=-1)
                        return logits
                    else:
                        logits_1 = self.classifier(query_rep_q2b[0])
                        logits_1 = logits_1.softmax(dim=-1)
                        logits_2 = self.classifier(query_rep_q2b[1])
                        logits_2 = logits_2.softmax(dim=-1)
                        
                        score = torch.max(torch.stack([logits_1, logits_2], dim=0), dim=0)[0]
                        return score


