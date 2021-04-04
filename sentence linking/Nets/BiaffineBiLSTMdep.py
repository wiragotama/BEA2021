"""
Author: Jan Wira Gotama Putra

We reference this implementation based on implementation by yzhangcs https://github.com/yzhangcs/biaffine-parser
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator
from allennlp.common.util import namespace_match
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.common import Params

from datasetreader import *
from Nets.BiaffineModules.mlp import MLP
from Nets.BiaffineModules.biaffine import Biaffine
from Nets.BiaffineModules.bilstm import BiLSTM
from Nets.BiaffineModules.dropout import SharedDropout

from treebuilder import TreeBuilder


class BiaffineBiLSTMdep(Model): 
    """
    Reimplementation of Biaffine BiLSTM model for dependency parsing
    """
    def __init__(self, vocab: Vocabulary, emb_dim: int, reduc_dim: int, lstm_u: int, n_stack: int, fc_u: int, n_dists: int, dropout_rate: float, torch_device: torch.device) -> None:
        """
        vocab (Vocabulary)
        emb_dim (int): sentence embedding dimension
        reduc_dim (int): the number of hidden layer for a dense layer to reduce the embedding dimension
        lstm_u (int): the number of lstm units
        n_stack(int): the number of BiLSTM stack
        fc_u (int): the number of hidden layer for the next dense layer after BiLSTM
        n_dists (int): the number of output distances
        dropout_rate (float): used for all dropouts: (1) sequence dropout, (2) dropout rate for between {BiLSTM and fc_u} and (3) between {fc_u and prediction}
        torch_device (torch.device): where this model supposed to run
        """
        super().__init__(vocab)
        
        self.reduc_dim = nn.Linear(emb_dim, reduc_dim) # input dimensionality reduction
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.bilstm = BiLSTM(input_size=reduc_dim,
                            hidden_size=lstm_u,
                            num_layers=n_stack,
                            dropout=dropout_rate)
        self.bilstm_dropout = SharedDropout(p=dropout_rate)

        # MLP layers, representing head (target) and dependent (source)
        self.mlp_arc_h = MLP(n_in=lstm_u*2,
                            n_hidden=fc_u,
                            dropout=dropout_rate)
        self.mlp_arc_d = MLP(n_in=lstm_u*2,
                            n_hidden=fc_u,
                            dropout=dropout_rate)

        # Biaffine layer
        self.arc_attn = Biaffine(n_in=fc_u,
                                bias_x=True,
                                bias_y=False)

        # loss function
        # each element in the prediction (s_arc), can be considered as a multi-class classification logits
        self.loss = nn.MultiMarginLoss(reduction='sum') # the original biaffine bilstm paper did not specify the loss function they used, and we follow Kipperwasser (2016) dependency parser to train using Max-Margin criterion

        # for saving model
        self.param = {
            "architecture": "BiaffineBiLSTMdep",
            "emb_dim": emb_dim,
            "reduc_dim": reduc_dim,
            "lstm_u": lstm_u,
            "n_stack": n_stack,
            "fc_u": fc_u,
            "n_dists": n_dists,
            "dropout_rate": dropout_rate
        }
        self.vocab = vocab
        self.torch_device = torch_device


    def __compute_loss(self, s_arc, rel_dists, seq_len): 
        """
        Compute loss (average of essay-level loss)

        Args:
            s_arc (torch.Tensor)
            rel_dists (torch.Tensor)
            seq_len (Any)

        Returns:
            (torch.Tensor, torch.Tensor)
        """
        def dist_idx_to_dist(idx):
            return int(self.vocab.get_token_from_index(int(idx), namespace="rel_dist_labels"))
        batch_size = len(rel_dists)

        # gold ans
        gold_ans = []
        for b in range(batch_size):
            non_padded_pred = rel_dists[b][:seq_len[b]].tolist()
            non_padded_pred = [dist_idx_to_dist(x) for x in non_padded_pred] 
            gold_matrix = torch.Tensor(TreeBuilder(non_padded_pred).adj_matrix)
            target = torch.argmax(gold_matrix, dim=-1) # index of the correct label
            if self.torch_device.type=="cuda": # move to device
                target = target.cuda()
            gold_ans.append(target)
            
        # pred ans
        pred_ans = []
        for b in range(batch_size):
            non_padded_pred = s_arc[b, :seq_len[b], :seq_len[b]]
            pred_ans.append(non_padded_pred) 

        # loss
        avg_loss = []
        for b in range(batch_size): # batch_size
            loss = self.loss(pred_ans[b], gold_ans[b]) # loss per essay
            avg_loss.append(loss) # loss per batch
        avg_loss = torch.mean(torch.stack(avg_loss))
        
        return pred_ans, avg_loss


    def forward(self, 
                sent_embeddings: torch.Tensor,
                rel_dists: torch.Tensor,
                seq_len: Any,
                essay_code: Any) -> Dict:
        """
        Forward passf
        
        Args:
            sent_embeddings (torch.Tensor): of size (batch_size, seq_len, emb_dim)
            rel_dists (torch.Tensor): of size (batch_size, seq_len, output_labels)
            seq_len (Any)
            essay_code (Any)

        Returns:
            Dict
        """
        inp_shape = sent_embeddings.shape # (batch_size, seq_len, embeddings)
        # print("sequence sizes", seq_len)

        # dimensionality reduction of sentence embedding
        flattened_embeddings = sent_embeddings.view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, embeddings)
        reduc_emb = F.leaky_relu(self.reduc_dim(flattened_embeddings), negative_slope=0.1) 
        reduc_emb = self.emb_dropout(reduc_emb) # we try to follow biaffine paper as much as possible, even though this part might be questionable. Is it good to have dropout on pre-trained embeddings?

        # prepare input for LSTM
        bilstm_inp = reduc_emb.view(inp_shape[0], inp_shape[1], self.param["reduc_dim"]) # (batch_size * seq_len, embeddings)
        bilstm_inp = pack_padded_sequence(bilstm_inp, torch.Tensor(seq_len), batch_first=True, enforce_sorted=False) # relu(0) = 0, so we can use padded_sequence here

        # BiLSTM
        bilstm_out, (hn, cn) = self.bilstm(bilstm_inp)
        bilstm_out, _ = pad_packed_sequence(bilstm_out, batch_first=True)

        # dropout after bilstm
        bilstm_out = self.bilstm_dropout(bilstm_out)

        # apply MLPs to BiLSTM output states
        arc_h = self.mlp_arc_h(bilstm_out) # head
        arc_d = self.mlp_arc_d(bilstm_out) # dependent

        # get arc scores from the bilinear attention
        s_arc = self.arc_attn(arc_d, arc_h)
    
        # loss
        pred_ans, loss = self.__compute_loss(s_arc, rel_dists, seq_len)

        # putput to user
        output = { "pred_matrix": pred_ans, 
                    "seq_len": seq_len,
                    "loss": loss}

        return output


