#pylint: disable=no-member
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from .wasp_n2v import NodeEmbedding

class BertForMultipleChoiceWasp(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)        
        self.use_n2v = config.use_n2v
        if self.use_n2v:
            n2v = NodeEmbedding(config.n2v_path)            
            self.n2v_emb = n2v.get_n2v_embedding_layer()            
            hidden_size = config.hidden_size + n2v.dim            
        else:
            hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        n2v_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):        
        num_choices = input_ids.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        n2v_ids = n2v_ids.view(-1, n2v_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None        

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        if self.use_n2v:
            n2v_weights = self.n2v_emb(n2v_ids)                        
            n2v_maxpool = nn.MaxPool1d(n2v_ids.shape[1])            
            n2v_vec = n2v_maxpool(n2v_weights.permute(0,2,1)).squeeze()            
            #pylint: disable=no-member            
            pooled_output = torch.cat([outputs[1], n2v_vec], axis=1)            
            
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
