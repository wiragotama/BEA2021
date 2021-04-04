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
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator
from allennlp.common.util import namespace_match
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.common import Params

from transformers import BertModel, DistilBertModel

from datasetreader import *
from Nets.BiLSTMparser import multi_task_dynamic_loss, multi_task_weighted_loss
from sentence_transformers import SentenceTransformer


class SBERT(SentenceTransformer):
    def __produce_embedding(self, features, output_value, convert_to_numpy, all_embeddings,):
        out_features = self.forward(features)
        embeddings = out_features[output_value]

        if output_value == 'token_embeddings':
            #Set token embeddings to 0 for padding tokens
            input_mask = out_features['attention_mask']
            input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * input_mask_expanded

        if convert_to_numpy:
            embeddings = embeddings.to('cpu').numpy()

        all_embeddings.extend(embeddings)

    """
    Overriding SentenceTransformer "encode" function so we can finetune SBERT
    """
    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None, output_value: str = 'sentence_embedding', convert_to_numpy: bool = True, requires_grad = True):
        """
        Computes sentence embeddings
        :param sentences:
           the sentences to embed
        :param batch_size:
           the batch size used for the computation
        :param show_progress_bar:
            Output a progress bar when encode sentences
        :param output_value:
            Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings
            to get wordpiece token embeddings.
        :param convert_to_numpy:
            If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param requires_grad:
            If true, requires_grad attribute of the output is True
        :return:
           Depending on convert_to_numpy, either a list of numpy vectors or a list of pytorch tensors
        """
        self.eval() # batchnorm or dropout layers will work in eval mode instead of training mode. It MIGHT not be good to have bn or dropout on BERT layer when training using a very small data, so keep self.eval() here, so we won't lose much information during the embedding process
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                tokens = self.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                features[feature_name] = torch.cat(features[feature_name]).to(self.device)

            if requires_grad == False: # no finetuning
                with torch.no_grad():
                    self.__produce_embedding(features, output_value, convert_to_numpy, all_embeddings)
            else: # finetuning
                self.__produce_embedding(features, output_value, convert_to_numpy, all_embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]

        return all_embeddings


class BertBiLSTMSTL(Model):
    """
    BERT + Stacked BiLSTM model for single task
    """
    def __init__(self, vocab: Vocabulary, emb_dim: int, lstm_u: int, n_stack: int, fc_u: int, n_dists: int, dropout_rate: float, torch_device: torch.device) -> None:
        """
        vocab (Vocabulary)
        emb_dim (int): sentence embedding dimension
        lstm_u (int): the number of lstm units
        n_stack(int): the number of BiLSTM stack
        fc_u (int): the number of hidden layer for the next dense layer after BiLSTM
        n_dists (int): the number of output distances
        dropout_rate (float): used for all dropouts: (1) sequence dropout, (2) dropout rate for between {BiLSTM and fc_u} and (3) between {fc_u and prediction}
        torch_device (torch.device): where this model supposed to run
        """
        super().__init__(vocab)
        
        # self.bert_layer = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.SBERT_layer = SBERT('bert-base-nli-mean-tokens')
        print("\n!!! Trial: use Sentence-BERT in BiLSTM model!!! \n")
        # self.bert_layer = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased') # TO DO: bert or distilbert

        # try finetuning using only the last three layers, TO DO: do the same for MTL
        # print("Trial: Only finetune last 3 layers of BERT / DistilBERT")
        # for name, param in self.bert_layer.named_parameters():
        #     if ("layer.11" in name) or ("layer.10" in name) or ("layer.9" in name):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.bilstm = nn.LSTM(input_size=emb_dim, 
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
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.prediction.weight)

        # for saving model
        self.param = {
            "architecture": "BertBiLSTMSTL",
            "emb_dim": emb_dim,
            "lstm_u": lstm_u,
            "n_stack": n_stack,
            "fc_u": fc_u,
            "n_dists": n_dists,
            "dropout_rate": dropout_rate
        }
        self.vocab = vocab
        self.torch_device = torch_device


    def forward(self, 
                sentences: torch.Tensor,
                rel_dists: torch.Tensor,
                seq_len: Any,
                essay_code: Any) -> Dict:
        """
        Forward passf
        
        Args:
            sentences (torch.Tensor): of size (batch_size, seq_len, n_tokens)
            rel_dists (torch.Tensor): of size (batch_size, seq_len, output_labels)
            seq_len (Any)
            essay_code (Any)

        Returns:
            Dict
        """
        # !!! TRIAL USING SBERT !!!
        inp_sentences = sentences
        inp_shape = (len(inp_sentences), len(inp_sentences[0])) # (batch_size, seq_len)

        # unpack batch to produce embeddings only for real sentences
        unpadded_batch = []
        for i in range(len(inp_sentences)):
            unpacked_input = deepcopy(inp_sentences[i][0:seq_len[i]])
            sent_embeddings = torch.stack(self.SBERT_layer.encode(unpacked_input, convert_to_numpy=False, requires_grad=True))
            unpadded_batch.append(sent_embeddings)
        # !!! END SBERT !!!

        # !!!!! if using BERT / DistilBERT
        # inp_sentences = sentences['tokens']
        # inp_shape = inp_sentences.shape # (batch_size, seq_len, n_tokens)

        # # unpack batch to produce embeddings only for real sentences
        # unpadded_batch = []
        # for i in range(len(inp_sentences)):
        #     unpacked_input = deepcopy(inp_sentences[i][0:seq_len[i]])
        #     attention_mask = SeqDatasetReader.get_essay_mask(unpacked_input)

        #     if self.torch_device.type=="cuda": # move to device
        #         unpacked_input = unpacked_input.cuda()  
        #         attention_mask = attention_mask.cuda()
        #     bert_outputs = self.bert_layer(unpacked_input, attention_mask)
        #     cont_reps = bert_outputs[0] # last hidden states
        #     sentence_embeddings = cont_reps[:, 0] # [CLS] vectors
        #     unpadded_batch.append(sentence_embeddings)
        # !!!!! END -- if using BERT / DistilBERT

        # pack batch for bilstm
        padded_batch = pad_sequence(unpadded_batch, batch_first=True, padding_value=0.0) # (batch_size, seq_len, emb_dim)
        bilstm_inp = padded_batch # (batch_size, seq_len, emb_dim)

        # BiLSTM
        bilstm_out, (hn, cn) = self.bilstm(bilstm_inp)

        # dense layer
        flattened_bilstm_out = bilstm_out.contiguous().view(inp_shape[0]*inp_shape[1], -1) # (batch_size * seq_len, hidden units)
        flattened_bilstm_out = self.dropout1(flattened_bilstm_out)
        dense_out = F.relu(self.fc(flattened_bilstm_out))

        # prediction
        dense_out = self.dropout2(dense_out)
        pred_logits = self.prediction(dense_out)
        pred_softmax = F.softmax(pred_logits, dim=-1)

        # reshape prediction to compute loss
        pred_logits = pred_logits.view(inp_shape[0], inp_shape[1], self.param["n_dists"])
        pred_softmax = pred_softmax.view(inp_shape[0], inp_shape[1], self.param["n_dists"])

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

        return output


class BertBiLSTMMTL(Model):
    """
    BERT + Stacked BiLSTM model for multi tasks (linking + node labelling)
    """
    def __init__(self, vocab: Vocabulary, emb_dim: int, lstm_u: int, n_stack: int, fc_u: int, n_dists: int, n_labels: int, dropout_rate: float, torch_device: torch.device, mtl_loss: str="weighted", weight_linking: float=0.5) -> None:
        """
        vocab (Vocabulary)
        emb_dim (int): sentence embedding dimension
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

        self.bert_layer = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bilstm = nn.LSTM(input_size=emb_dim, 
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
            "architecture": "BertBiLSTMMTL",
            "emb_dim": emb_dim,
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

    # TO DO: fix this problem
    def forward(self, 
                sentences: torch.Tensor,
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
        inp_sentences = sentences['tokens']
        inp_shape = inp_sentences.shape # (batch_size, seq_len, n_tokens)

        # prepare input mask
        token_mask = SeqDatasetReader.get_text_field_mask(inp_sentences)

        # flatten the input
        flattened_input = inp_sentences.view(inp_shape[0] * inp_shape[1], -1) # (batch_size * seq_len, n_tokens)
        flattened_token_mask = token_mask.view(inp_shape[0] * inp_shape[1], -1) # (batch_size * seq_len, n_tokens)

        # forward to BERT
        if self.torch_device.type=="cuda": # move to device
            flattened_token_mask = flattened_token_mask.cuda()        
        bert_outputs = self.bert_layer(flattened_input,         # token converted as index
                                        flattened_token_mask    # attention mask
                                    )        
        cont_reps = bert_outputs[0] # last hidden states
        # get CLS vector and classify (sentence embeddings)
        sentence_embeddings = cont_reps[:, 0]

         # prepare input for bilstm
        bilstm_inp = sentence_embeddings.view(inp_shape[0], inp_shape[1], self.param["emb_dim"]) # (batch_size, seq_len, emb_dim)

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


