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


class ConcatPairClassifier(Model):
    """
    Reduce the dimension of source and target sentence embedding, then concat them, then perform classification
    """
    def __init__(self, vocab: Vocabulary, emb_dim: int, fc1_u: int, fc2_u: int, dropout_rate: float, n_labels: int) -> None:
        """
        vocab (Vocabulary)
        emb_dim (int): sentence embedding dimension
        fc1_u (int): the number of hidden layer for fc1
        fc2_u (int): the number of hidden layer for fc2
        dropout_rate (float): dropout rate between fc2 and fc_3
        n_labels (int): the number of labels
        """
        super().__init__(vocab)
        self.fc1_1 = nn.Linear(emb_dim, fc1_u) # input dimensionality reduction
        self.fc1_2 = nn.Linear(emb_dim, fc1_u) # input dimensionality reduction
        self.fc2 = nn.Linear(fc1_u * 2, fc2_u)
        self.dropout = nn.Dropout(dropout_rate) # will be disabled in model.eval()
        self.fc3 = nn.Linear(fc2_u, n_labels)

        # weight initialization, http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # xavier is good for FNN
        torch.nn.init.xavier_uniform_(self.fc1_1.weight)
        torch.nn.init.xavier_uniform_(self.fc1_2.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.loss = nn.NLLLoss()

        # for saving model
        self.param = {
            "class": "ConcatPairClassifier",
            "emb_dim": emb_dim,
            "fc1_u": fc1_u,
            "fc2_u": fc2_u,
            "dropout_rate": dropout_rate,
            "n_labels": n_labels
        }
        self.vocab = vocab
        

    def forward(self, 
            source_emb: torch.Tensor,
            target_emb: torch.Tensor,
            label: torch.Tensor,
            essay_code: Any) -> Dict:
        """
        Forward pass
        """
        assert (source_emb.shape == target_emb.shape)
        x1 = F.relu(self.fc1_1(source_emb))
        x2 = F.relu(self.fc1_2(target_emb))

        # combine
        x = torch.cat((x1, x2), dim=1)

        # classify
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x), dim=-1)

        # loss calculation
        output = {"class_log_softmax": x}
        output["loss"] = self.loss(x, label)

        return output


class LSTMPairClassifier(Model):
    """
    Change the "concat" part of ConcatPairClassifier using LSTM
    """
    def __init__(self, vocab: Vocabulary, emb_dim: int, fc1_u: int, fc2_u: int, lstm_u: int, dropout_rate: float, n_labels: int) -> None:
        """
        vocab (Vocabulary)
        emb_dim (int): sentence embedding dimension
        fc1_u (int): the number of hidden layer for fc1
        fc2_u (int): the number of hidden layer for fc2
        lstm_u (int): the number of hidden layer for LSTM
        dropout_rate (float): dropout rate between fc2 and fc_3
        n_labels (int): the number of labels
        """
        super().__init__(vocab)
        self.fc1_1 = nn.Linear(emb_dim, fc1_u) # input dimensionality reduction
        self.fc1_2 = nn.Linear(emb_dim, fc1_u) # input dimensionality reduction
        self.lstm = nn.LSTM(input_size=fc1_u, hidden_size=lstm_u, batch_first=True)
        self.fc2 = nn.Linear(lstm_u * 2, fc2_u)
        self.dropout = nn.Dropout(dropout_rate) # will be disabled in model.eval()
        self.fc3 = nn.Linear(fc2_u, n_labels)

        # weight initialization, http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # xavier is good for FNN
        torch.nn.init.xavier_uniform_(self.fc1_1.weight)
        torch.nn.init.xavier_uniform_(self.fc1_2.weight)
        # lstm initialization is left to default
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.loss = nn.NLLLoss()

        # for saving model
        self.param = {
            "class": "LSTMPairClassifier",
            "emb_dim": emb_dim,
            "fc1_u": fc1_u,
            "lstm_u": lstm_u,
            "fc2_u": fc2_u,
            "dropout_rate": dropout_rate,
            "n_labels": n_labels
        }
        self.vocab = vocab
        

    def forward(self, 
            source_emb: torch.Tensor,
            target_emb: torch.Tensor,
            label: torch.Tensor,
            essay_code: Any) -> torch.Tensor:
        """
        Forward pass
        """
        assert (source_emb.shape == target_emb.shape)
        batch_size = source_emb.shape[0]
        x1 = F.relu(self.fc1_1(source_emb))
        x2 = F.relu(self.fc1_2(target_emb))

        # combine
        x = torch.cat((x1, x2), dim=1)
        x = x.view(batch_size, 2, self.param["fc1_u"])

        # put source -> target to lstm
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_output_cat = lstm_out.contiguous().view(batch_size, -1)

        # classify
        x = F.relu(self.fc2(lstm_output_cat))
        x = self.dropout(x)
        x = F.log_softmax(self.fc3(x), dim=-1)

        # loss calculation
        output = {"class_log_softmax": x}
        output["loss"] = self.loss(x, label)

        return output