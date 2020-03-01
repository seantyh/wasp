#pylint: disable=no-member
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from .wasp_n2v import NodeEmbedding
from .utils import get_data_path

N2V_PATH = str(get_data_path("sem_graph", "node2vec_sem_graph.pkl"))
class BertForMultipleChoiceWasp(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        # [NOTE] due to config is only persist as BertConfig, 
        # config.use_n2v are only available in training time.
        # Manually change the default boolean value here for switching on use_n2v
        self.use_n2v = config.use_n2v if hasattr(config, "use_n2v") else False
        if self.use_n2v:
            n2v_path = config.n2v_path if hasattr(config, "n2v_path") else N2V_PATH
            n2v = NodeEmbedding(n2v_path)
            self.n2v_dim = n2v.dim
            self.n2v_emb = n2v.get_n2v_embedding_layer()
            self.n2v_hidden = nn.Sequential(nn.Linear(n2v.dim, n2v.dim), nn.ReLU())
            self.n2v_rnn = nn.GRU(n2v.dim, n2v.dim, batch_first=True)
            hidden_size = config.hidden_size + n2v.dim
        else:
            hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_rnn_hidden(self, batch_size):
        # 4 options are reshaped into first dimension
        hidden = torch.rand(1, batch_size*4, self.n2v_dim)
        return hidden

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
        rnn_hidden=None
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
            # n2v_pool = nn.AvgPool1d(n2v_ids.shape[1])
            # n2v_vec = n2v_pool(n2v_weights.permute(0,2,1)).squeeze()            
            n2v_out, rnn_hidden = self.n2v_rnn(n2v_weights, rnn_hidden)
            n2v_vec = rnn_hidden.squeeze()

            n2v_h = self.n2v_hidden(n2v_vec)
            #pylint: disable=no-member
            pooled_output = torch.cat([outputs[1], n2v_h], axis=1)

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
