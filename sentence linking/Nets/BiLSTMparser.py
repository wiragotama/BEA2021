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
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.common import Params

from datasetreader import *


class BiLSTMSTL(Model):
    """
    Stacked BiLSTM model for single task
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
        self.bilstm = nn.LSTM(input_size=reduc_dim, 
                            hidden_size=lstm_u,
                            num_layers=n_stack,
                            batch_first=True,
                            bidirectional=True) # no dropout between bilstm layers
        self.dropout1 = nn.Dropout(dropout_rate) # will be disabled in model.eval
        self.fc = nn.Linear(lstm_u * 2, fc_u)
        self.dropout2 = nn.Dropout(dropout_rate) # will be disabled in model.eval()
        self.prediction = nn.Linear(fc_u, n_dists)

        # weight initialization, http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # xavier is good for FNN
        torch.nn.init.xavier_uniform_(self.reduc_dim.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.prediction.weight)

        # for saving model
        self.param = {
            "architecture": "BiLSTMSTL",
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
        reduc_emb = F.relu(self.reduc_dim(flattened_embeddings)) 
        # print("embeddings", sent_embeddings.shape)
        # print("flattened", flattened_embeddings.shape)
        # print("embeddings (dim reduction)", reduc_emb.shape)

        # prepare input for LSTM
        bilstm_inp = reduc_emb.view(inp_shape[0], inp_shape[1], self.param["reduc_dim"])
        # print("bilstm input", bilstm_inp.shape)

        # BiLSTM
        bilstm_out, (hn, cn) = self.bilstm(bilstm_inp)
        # print("bilstm output", bilstm_out.shape)

        # dense layer
        flattened_bilstm_out = bilstm_out.contiguous().view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, hidden units)
        flattened_bilstm_out = self.dropout1(flattened_bilstm_out)
        dense_out = F.relu(self.fc(flattened_bilstm_out))
        # print("flattened bilstm out", flattened_bilstm_out.shape)
        # print(flattened_bilstm_out)
        # print("dense layer", dense_out.shape)

        # prediction
        dense_out = self.dropout2(dense_out)
        pred_logits = self.prediction(dense_out)
        pred_softmax = F.softmax(pred_logits, dim=-1)

        # reshape prediction to compute loss
        pred_logits = pred_logits.view(inp_shape[0], inp_shape[1], self.param["n_dists"])
        pred_softmax = pred_softmax.view(inp_shape[0], inp_shape[1], self.param["n_dists"])
        # print("pred_logits", pred_logits.shape)
        # print("pred_softmax", pred_softmax.shape)

        # sequence masking to compute loss
        mask = SeqDatasetReader.get_batch_seq_mask(seq_len)
        if self.torch_device.type=="cuda": # move to device
            mask = mask.cuda()   

        # loss
        loss = sequence_cross_entropy_with_logits(pred_logits, rel_dists, mask)

        # putput to user
        output = { "pred_logits": pred_logits, 
                    "pred_softmax" : pred_softmax,
                    "seq_mask": tonp(mask),
                    "loss": loss}
        # print("loss", loss)

        return output


class BiLSTMMTL(Model):
    """
    Stacked BiLSTM model for multi tasks (linking + node labelling)
    """
    def __init__(self, vocab: Vocabulary, emb_dim: int, reduc_dim: int, lstm_u: int, n_stack: int, fc_u: int, n_dists: int, n_labels: int, dropout_rate: float, torch_device: torch.device, mtl_loss: str="weighted", weight_linking: float=0.5) -> None:
        """
        vocab (Vocabulary)
        emb_dim (int): sentence embedding dimension
        reduc_dim (int): the number of hidden layer for a dense layer to reduce the embedding dimension
        lstm_u (int): the number of lstm units
        n_stack(int): the number of BiLSTM stack
        fc_u (int): the number of hidden layer for the next dense layer after BiLSTM
        n_dists (int): the number of output distances (linking)
        n_labels (int): the number of component labels (node labelling)
        dropout_rate (float): used for all dropouts: (1) sequence dropout, (2) dropout rate for between {BiLSTM and fc_u} and (3) between {fc_u and prediction}
        torch_device (torch.device): where this model supposed to run
        mtl_loss (str): how to combine mtl loss {"average", "weighted", "dynamic"}
        weight_linking (str): only used for "weighted" loss, default=0.5

        """
        if mtl_loss not in {"average", "weighted", "dynamic"}:
            raise Exception('mtl_loss options: {"average", "weighted", "dynamic"}')
        if mtl_loss=="weighted" and (weight_linking < 0.0 or weight_linking > 1.0):
            raise Exception('weight_linking must be between [0.0, 1.0]')

        super().__init__(vocab)

        self.reduc_dim = nn.Linear(emb_dim, reduc_dim) # input dimensionality reduction
        self.bilstm = nn.LSTM(input_size=reduc_dim, 
                            hidden_size=lstm_u,
                            num_layers=n_stack,
                            batch_first=True,
                            bidirectional=True) # no dropout between bilstm layers
        self.dropout1 = nn.Dropout(dropout_rate) # will be disabled in model.eval
        self.fc = nn.Linear(lstm_u * 2, fc_u)
        self.dropout2 = nn.Dropout(dropout_rate) # will be disabled in model.eval()
        self.prediction_linking = nn.Linear(fc_u, n_dists) # linking
        self.prediction_node_labelling = nn.Linear(fc_u, n_labels) # node labellling

        # weight initialization, http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # xavier is good for FNN
        torch.nn.init.xavier_uniform_(self.reduc_dim.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.prediction_linking.weight)
        torch.nn.init.xavier_uniform_(self.prediction_node_labelling.weight)

        # set loss
        self.mtl_loss = mtl_loss
        if self.mtl_loss == "weighted":
            self.weight_linking = weight_linking
        else:
            self.weight_linking = 0.5

        # for dynamically weight the loss in MTL
        if self.mtl_loss == "dynamic":
            self.eta_A = nn.Parameter(torch.zeros(1)) # defined as 2*log(sigma) https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681/12
            self.eta_B = nn.Parameter(torch.zeros(1)) # sigma is initialized as 1, that's why 2*log(sigma)=0

        # for saving model
        self.param = {
            "architecture": "BiLSTMMTL",
            "emb_dim": emb_dim,
            "reduc_dim": reduc_dim,
            "lstm_u": lstm_u,
            "n_stack": n_stack,
            "fc_u": fc_u,
            "n_dists": n_dists,
            "n_labels": n_labels,
            "dropout_rate": dropout_rate,
            "mtl_loss": mtl_loss,
            "weight_linking": weight_linking
        }
        self.vocab = vocab
        self.torch_device = torch_device


    def forward(self, 
                sent_embeddings: torch.Tensor,
                rel_dists: torch.Tensor,
                component_labels: torch.Tensor,
                seq_len: Any,
                essay_code: Any) -> Dict:
        """
        Forward passf
        
        Args:
            sent_embeddings (torch.Tensor): of size (batch_size, seq_len, emb_dim)
            rel_dists (torch.Tensor): of size (batch_size, seq_len, output_labels)
            component_labels (torch.Tensor): of size (batch_size, seq_len, output_labels)
            seq_len (Any)
            essay_code (Any)

        Returns:
            Dict
        """
        inp_shape = sent_embeddings.shape # (batch_size, seq_len, embeddings)

        # dimensionality reduction of sentence embedding
        flattened_embeddings = sent_embeddings.view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, embeddings)
        reduc_emb = F.relu(self.reduc_dim(flattened_embeddings)) 
        # prepare input for LSTM
        bilstm_inp = reduc_emb.view(inp_shape[0], inp_shape[1], self.param["reduc_dim"])

        # BiLSTM
        bilstm_out, (hn, cn) = self.bilstm(bilstm_inp)

        # dense layer
        flattened_bilstm_out = bilstm_out.contiguous().view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, hidden units)
        flattened_bilstm_out = self.dropout1(flattened_bilstm_out)
        dense_out = F.relu(self.fc(flattened_bilstm_out))

        # prediction for LINKING
        dense_out = self.dropout2(dense_out)
        pred_linking_logits = self.prediction_linking(dense_out)
        pred_linking_softmax = F.softmax(pred_linking_logits, dim=-1)
        # reshape prediction to compute loss
        pred_linking_logits = pred_linking_logits.view(inp_shape[0], inp_shape[1], self.param["n_dists"])
        pred_linking_softmax = pred_linking_softmax.view(inp_shape[0], inp_shape[1], self.param["n_dists"])


        # prediction for NODE LABELLING
        dense_out = self.dropout2(dense_out)
        pred_node_labelling_logits = self.prediction_node_labelling(dense_out)
        pred_node_labelling_softmax = F.softmax(pred_node_labelling_logits, dim=-1)
        # reshape prediction to compute loss
        pred_node_labelling_logits = pred_node_labelling_logits.view(inp_shape[0], inp_shape[1], self.param["n_labels"])
        pred_node_labelling_softmax = pred_node_labelling_softmax.view(inp_shape[0], inp_shape[1], self.param["n_labels"])


        # sequence masking to compute loss
        mask = SeqDatasetReader.get_batch_seq_mask(seq_len)
        if self.torch_device.type=="cuda": # move to device
            mask = mask.cuda()   

        # weighted MTL loss
        if self.mtl_loss == "average" or self.mtl_loss == "weighted":
            loss = multi_task_weighted_loss(pred_linking_logits, rel_dists, pred_node_labelling_logits, component_labels, mask, self.weight_linking)
        elif self.mtl_loss == "dynamic":
            loss = multi_task_dynamic_loss(pred_linking_logits, rel_dists, pred_node_labelling_logits, component_labels, mask, self.eta_A, self.eta_B)

        # putput to user
        output = { "pred_linking_logits": pred_linking_logits, 
                    "pred_linking_softmax" : pred_linking_softmax,
                    "pred_node_labelling_logits": pred_node_labelling_logits,
                    "pred_node_labelling_softmax": pred_node_labelling_softmax,
                    "seq_mask": tonp(mask),
                    "loss": loss}

        return output


def multi_task_weighted_loss(pred_linking_logits, rel_dists, pred_node_labelling_logits, component_labels, mask, weight_linking) -> torch.Tensor:
    """
    Compute weighted loss

    Args:
        pred_linking_logits (torch.Tensor)
        rel_dists (torch.Tensor)
        pred_node_labelling_logits (torch.Tensor)
        component_labels (torch.Tensor)
        eta_A (torch.Tensor): parameter for weighting the first task's loss
        eta_B (torch.Tensor): parameter for weighting the second task's loss

    Returns:
        loss (torch.Tensor)
    """
    if weight_linking < 0.0 or weight_linking > 1.0:
        raise Exception("weight_linking should be between [0.0, 1.0]")

    loss_linking = sequence_cross_entropy_with_logits(pred_linking_logits, rel_dists, mask)
    loss_node_labelling = sequence_cross_entropy_with_logits(pred_node_labelling_logits, component_labels, mask)

    return weight_linking * loss_linking + (1.0-weight_linking) * loss_node_labelling


def multi_task_dynamic_loss(pred_linking_logits, rel_dists, pred_node_labelling_logits, component_labels, mask, eta_A, eta_B) -> torch.Tensor:
        """
        Compute dynamic weighting of task-specific losses during training process, based on homoscedastic uncertainty of tasks
        proposed by Kendall et al. (2018): multi task learning using uncertainty to weigh losses for scene geometry and semantics
        
        References:
        - https://arxiv.org/abs/1705.07115
        - https://github.com/ranandalon/mtl
        - https://discuss.pytorch.org/t/how-to-learn-the-weights-between-two-losses/39681
        - https://github.com/CubiCasa/CubiCasa5k/blob/master/floortrans/losses/uncertainty_loss.py

        Args:
            pred_linking_logits (torch.Tensor)
            rel_dists (torch.Tensor)
            pred_node_labelling_logits (torch.Tensor)
            component_labels (torch.Tensor)
            eta_A (torch.Tensor): parameter for weighting the first task's loss
            eta_B (torch.Tensor): parameter for weighting the second task's loss

        Returns:
            loss (torch.Tensor)
        """

        def weighted_loss(loss, eta):
            # print(loss.shape)
            # input()
            return torch.exp(-eta) * loss + torch.log(1+torch.exp(eta))

        loss_linking = sequence_cross_entropy_with_logits(pred_linking_logits, rel_dists, mask, average=None)
        loss_node_labelling = sequence_cross_entropy_with_logits(pred_node_labelling_logits, component_labels, mask, average=None)

        loss = weighted_loss(loss_linking, eta_A) + weighted_loss(loss_node_labelling, eta_B)
        return torch.mean(loss)


