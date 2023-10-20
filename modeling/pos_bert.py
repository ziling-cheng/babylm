import torch.nn as nn

from transformers import BertModel, BertForMaskedLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)

from .pos_tokenizer import POSTokenizer


class BertForMaskedLMWithPOSEmb(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def get_pos_tag_embeddings(self):
        return self.pos_embedding

    def set_pos_tag_embeddings(self, value):
        self.pos_embedding = value

    def forward(
        self,
        input_ids=None,
        pos_tag_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        input_embeddings = self.bert.embeddings(input_ids)
        pos_embeddings = self.pos_embedding(pos_tag_ids)
        input_embeddings = input_embeddings + pos_embeddings
        return super().forward(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs,
        )

    def init_weights(self):
        super().init_weights()
        # we are initializing the position embedding here instead of in __init__ because
        # super.__init__ calls post_init which in turn calls init_weights and in that case
        # we will get an error since the pos_embedding is not yet defined and we are trying
        # to initialize it's weights
        self.pos_embedding = nn.Embedding(len(POSTokenizer.POS_TAGS), self.config.hidden_size)
        self.pos_embedding.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
