"""
Author: Jan Wira Gotama Putra
"""
from typing import *
from tqdm import tqdm
import time
import numpy as np
import os
import json
import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator
from allennlp.common.util import namespace_match
from allennlp.nn.util import get_text_field_mask

from transformers import BertModel, DistilBertModel


class BERTFinetuning(Model):

    def __init__(self, vocab: Vocabulary, n_labels: int, torch_device: torch.device) -> None:
        """
        Args:
            vocab (Vocabulary)
            fc_u (int): the number of units of hidden layer for classification
            dropout_rate (float)
            n_labels (int): the number of labels
            torch_device (torch.device): device to use
        """
        super().__init__(vocab)
        self.emb_dim = 768
        self.bert_layer = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.fc = nn.Linear(self.emb_dim, n_labels)

        # weight initialization, http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # xavier is good for FNN
        torch.nn.init.xavier_uniform_(self.fc.weight)

        # loss
        self.loss = nn.NLLLoss()

        # for saving model
        self.param = {
            "class": "BERTFinetuning",
            "emb_dim": self.emb_dim,
            "n_labels": n_labels
        }
        self.vocab = vocab
        self.running_device = torch_device
        

    def forward(self, 
            source_target_sent: Dict[str, torch.Tensor],
            label: torch.Tensor,
            essay_code: Any) -> torch.Tensor:
        """
        Forward pass
        """
        def get_text_field_mask(tokens: torch.Tensor) -> torch.Tensor:
            """
            The attention mask length does not match with wordpiece tokenizer, so this is the solution

            Args:
                tokens (Tensor): of (batch_size, dimension); text already split and then converted to their indices
            """
            masks = []
            for token in tokens:
                mask = []
                for e in token:
                    if e.item() == 0: # padding
                        mask.append(0)
                    else:
                        mask.append(1)
                masks.append(mask)
            return torch.LongTensor(np.array(masks))

        attn_mask = get_text_field_mask(source_target_sent['tokens']) # allennlp mask dimension is different to those of wordpiece, https://github.com/allenai/allennlp/issues/2668
        if self.running_device.type=="cuda": # move to device
            attn_mask = attn_mask.cuda()        
        outputs = self.bert_layer(source_target_sent['tokens'], # token converted as index
                                    attn_mask,                              # attention mask
                                    source_target_sent['tokens-type-ids'])  # segment id
        cont_reps = outputs[0] # last hidden states
        
        # get CLS vector and classify
        cls_rep = cont_reps[:, 0]
        x = F.log_softmax(self.fc(cls_rep), dim=-1)

        # loss calculation
        output = {"class_log_softmax": x}
        output["loss"] = self.loss(x, label)

        return output


class DistilBERTFinetuning(Model):

    def __init__(self, vocab: Vocabulary, n_labels: int, torch_device: torch.device) -> None:
        """
        Args:
            vocab (Vocabulary)
            fc_u (int): the number of units of hidden layer for classification
            dropout_rate (float)
            n_labels (int): the number of labels
            torch_device (torch.device): device to use
        """
        super().__init__(vocab)
        self.emb_dim = 768
        self.distil_bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        self.fc = nn.Linear(self.emb_dim, n_labels)

        # weight initialization, http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # xavier is good for FNN
        torch.nn.init.xavier_uniform_(self.fc.weight)

        # loss
        self.loss = nn.NLLLoss()

        # for saving model
        self.param = {
            "class": "DistilBERTFinetuning",
            "emb_dim": self.emb_dim,
            "n_labels": n_labels
        }
        self.vocab = vocab
        self.running_device = torch_device
        

    def forward(self, 
            source_target_sent: Dict[str, torch.Tensor],
            label: torch.Tensor,
            essay_code: Any) -> torch.Tensor:
        """
        Forward pass
        """
        def get_text_field_mask(tokens: torch.Tensor) -> torch.Tensor:
            """
            The attention mask length does not match with wordpiece tokenizer, so this is the solution

            Args:
                tokens (Tensor): of (batch_size, dimension); text already split and then converted to their indices
            """
            masks = []
            for token in tokens:
                mask = []
                for e in token:
                    if e.item() == 0: # padding
                        mask.append(0)
                    else:
                        mask.append(1)
                masks.append(mask)
            return torch.LongTensor(np.array(masks))

        attn_mask = get_text_field_mask(source_target_sent['tokens']) # allennlp mask dimension is different to those of wordpiece, https://github.com/allenai/allennlp/issues/2668
        if self.running_device.type=="cuda": # move to device
            attn_mask = attn_mask.cuda()        
        outputs = self.distil_bert(input_ids=source_target_sent['tokens'], # token converted as index
                                        attention_mask=attn_mask)               # attention mask
                                        # distil bert does not require token-type-ids
        cont_reps = outputs[0] # last hidden states
        
        # get CLS vector and classify
        cls_rep = cont_reps[:, 0]
        x = F.log_softmax(self.fc(cls_rep), dim=-1)

        # loss calculation
        output = {"class_log_softmax": x}
        output["loss"] = self.loss(x, label)

        return output