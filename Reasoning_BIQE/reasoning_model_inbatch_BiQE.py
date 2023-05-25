import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


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

    def forward(self, batch, is_prepare=False, candidate_rep=None, schema='matching'):
        if is_prepare:
            candidates_rep = self.encode_seq(batch['entity_input_ids'], batch['entity_masks'], batch.get('entity_type_ids', None))[0]
            return candidates_rep
        else:
            query_rep, all_query_rep = self.encode_seq(batch['query_input_ids'], batch['query_mask'], batch.get('query_type_ids', None), position_ids=batch['position_ids']) #, position_ids=batch['position_ids']

            mask_index = batch["mask_positions"].unsqueeze(-1).expand(-1, -1, all_query_rep.size(-1))
            mask_rep = torch.gather(all_query_rep, 1, mask_index)

            logits = self.classifier(mask_rep)
            score = logits.softmax(dim=-1)
            select_mask = (batch["mask_positions"] != 0).int().unsqueeze(-1)
            
            score = torch.sum(score * select_mask, dim=1)/torch.sum(select_mask, 1)

            if self.training:
                if schema == 'matching':
                    query_mask_rep = torch.sum(mask_rep * select_mask, dim=1)/torch.sum(select_mask, 1)
                    ans_rep = self.encode_seq(batch['ans_input_ids'], batch['ans_masks'], batch.get('a_type_ids', None))[0]

                    neg_ans_rep = ans_rep[batch["negative_index"]] 
                    query_mask_rep = query_mask_rep.unsqueeze(1)
                    neg_ans_rep = neg_ans_rep.transpose(-2, -1)
                    logits = query_mask_rep @ neg_ans_rep
                    logits = logits.squeeze(1)

                    cls_loss_fn = nn.CrossEntropyLoss()
                    cls_loss = cls_loss_fn(
                        logits/1.0,
                        batch['tags'].squeeze()
                    )
                    return cls_loss
                else:
                    classification_loss = self.cross_entropy(
                        score,
                        batch['selected_ans'].squeeze()
                    )
                    return classification_loss
            else:
                if schema == 'matching':
                    query_mask_rep = query_mask_rep.unsqueeze(1)
                    neg_ans_rep = candidate_rep.unsqueeze(0).repeat(query_mask_rep.size(0), 1, 1).transpose(-2, -1)
                    logits = (query_mask_rep @ neg_ans_rep).squeeze(1) / 1.0
                    logits = logits.softmax(dim=-1)
                    return logits
                else:
                    return score




