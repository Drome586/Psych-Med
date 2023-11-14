import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

import torch.nn.functional as F

from pytorch_transformers import BertPreTrainedModel, BertModel

class BertForSequenceClassificationWithAttention(BertPreTrainedModel):
    """把bert最后的输出改为attention"""
    def __init__(self, config):
        super(BertForSequenceClassificationWithAttention, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))
        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        last_hidden_states = outputs[0]

        last_hidden_states = self.dropout(last_hidden_states)
        
        score= torch.tanh(torch.matmul(last_hidden_states, self.W_w))
        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        scored_x = last_hidden_states * attention_weights
        feat = torch.sum(scored_x, dim=1)
        
        logits = self.classifier(feat)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    
class BertForSequenceClassificationWithLast3layersAttention(BertPreTrainedModel):
    """把bert最后的输出改为attention"""
    def __init__(self, config):
        super(BertForSequenceClassificationWithLast3layersAttention, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.W_w1 = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.W_w2 = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.W_w3 = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.W_w1, -0.1, 0.1)
        nn.init.uniform_(self.W_w2, -0.1, 0.1)
        nn.init.uniform_(self.W_w3, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        all_hidden_states = outputs[2]
        
        merge_last3layers = torch.selu(torch.matmul(all_hidden_states[-3], self.W_w1) + torch.matmul(all_hidden_states[-2], self.W_w2) + torch.matmul(all_hidden_states[-1], self.W_w3))

        merge_last3layers = self.dropout(merge_last3layers)
        
        score= torch.tanh(torch.matmul(merge_last3layers, self.W_w))
        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        scored_x = merge_last3layers * attention_weights
        feat = torch.sum(scored_x, dim=1)
        
        logits = self.classifier(feat)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
    
class BertForSequenceClassificationWithLast3layersPooler(BertPreTrainedModel):
    """把bert最后的输出改为attention"""
    def __init__(self, config):
        super(BertForSequenceClassificationWithLast3layersPooler, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.W_w1 = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.W_w2 = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.W_w3 = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        nn.init.uniform_(self.W_w1, -0.1, 0.1)
        nn.init.uniform_(self.W_w2, -0.1, 0.1)
        nn.init.uniform_(self.W_w3, -0.1, 0.1)
        
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        all_hidden_states = outputs[2]
        
        merge_last3layers = torch.selu(torch.matmul(all_hidden_states[-3], self.W_w1) + torch.matmul(all_hidden_states[-2], self.W_w2) + torch.matmul(all_hidden_states[-1], self.W_w3))

        merge_last3layers = self.dropout(merge_last3layers)
        
        feat = self.pooler(merge_last3layers)
        
        logits = self.classifier(feat)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)