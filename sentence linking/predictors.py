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
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.common.util import namespace_match
from allennlp.nn import util as nn_util

from Nets.BiLSTMparser import *
from model_functions import *
from datasetreader import tonp
from treebuilder import TreeBuilder
from EdmondMST import MSTT


def get_rank_order(attn_matrix):
    """
    Args:
        attn_matrix (numpy.ndarray): a matrix where element [i,j] denotes the probability (or score) of sentence-i points to sentence-j (j as a target)

    Returns:
        rank_order (list[list]): we have the rank of each node per row, meaning how likely sentence-i points to sentence-j (in temrs of ranking)
    """

    idx = np.argsort((-attn_matrix), axis=-1) # gives the index of the largest element, e.g., idx[0] is the index of the 0-th largest element 

    # adjust the idx matrix, so we now have the rank of each node (per row), this is easier for debugging
    rank_order = np.zeros((len(idx), len(idx)), dtype=int)
    for i in range(len(idx)):
        for j in range(len(idx[i])):
            rank_order[i, idx[i,j]] = int(j)

    return rank_order


def run_MST(rank_order, weight_matrix, verdict="min"):
    """
    Perform an MST algorithm on the attn_matrix (directed graph)

    Args:
        rank_order (list[list]): how likely sentence-i points to sentence-j (in temrs of ranking)
        weight_matrix (numpy.ndarray): matrix containing the weight of edges 
        verdict (str): minimum ("min") or maximum ("max") spanning tree

    Returns:
        dist_interpretation (list): the distance between each node to its target sentence
    """
    diag = np.diag(rank_order)
    root_candidate = np.argmin(diag) # the most probable node that points to itself
    min_val = min(diag) # the weight (rank order) of a node points to itself
    non_ACs = set()
    if min_val == 0: # there might be multiple plausible root candidates, but we only use the first node (the node that appears the first in the text) and regards the rest as non-ACS
        for i in range(len(diag)):
            if diag[i] == min_val and i!=root_candidate:
                non_ACs.add(i) 

    # list of edges for MST
    MST = MSTT()
    for i in range(len(weight_matrix)):
        for j in range(len(weight_matrix)):
            if not(i in non_ACs) and not(j in non_ACs):
                MST.add_edge(i, weight_matrix[i,j], j) # source, weight, target

    # run MST algorithm
    if verdict == "min":
        mst_arcs = MST.min_spanning_arborescence(root_candidate)
    else:
        mst_arcs = MST.max_spanning_arborescence(root_candidate)

    # distance interpretation
    dist_interpretation = [0] * len(rank_order)
    for arc in mst_arcs:
        dist_interpretation[arc.tail] = arc.head - arc.tail
    
    return dist_interpretation


class PredictorSTL:
    def __init__(self, 
                model: Model, 
                iterator: DataIterator,
                cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device


    def dist_idx_to_dist(self, idx):
        """
        Args:
            idx (int): index of rel_dist_labels
        """
        return int(self.model.vocab.get_token_from_index(int(idx), namespace="rel_dist_labels"))


    def _unpack_model_batch_prediction(self, batch, coerce_tree=False) -> np.ndarray:
        """
        Interpret prediction result per batch
        """
        out_dict = self.model(**batch)
        pred_softmax = tonp(out_dict["pred_softmax"])
        # print("seq len", batch["seq_len"])
        # print(pred_softmax.shape)

        batch_interpretation = []
        for es in range(len(pred_softmax)):
        
            essay_interpretation = []
            max_seq_len = batch["seq_len"][es]

            # simple decoding using argmax
            for s in range(max_seq_len): # iterate each sentence in the essay, s is the index of the current sentence
                curr_pred = pred_softmax[es][s]

                # perform constrained argmax
                ranked_pred = [i for i in reversed(sorted(enumerate(curr_pred), key=lambda x:x[1]))]
                # print(ranked_pred)
                for i in range(len(ranked_pred)):
                    tmp_dist = self.dist_idx_to_dist(ranked_pred[i][0])
                    # print(tmp_dist, tmp_dist+s)
                    # input()
                    if 0 <= tmp_dist + s <= max_seq_len-1:
                        pred_dist = tmp_dist
                        break

                essay_interpretation.append(pred_dist)

            # check if the output is tree
            rep = TreeBuilder(essay_interpretation)
            if (not rep.is_tree()) and (coerce_tree==True):
                attn_matrix = [] # element [i,j] denotes the probability of sentence i connects to sentence j (j as the target)
                for s in range(max_seq_len): # iterate each sentence in the essay, s is the index of the current sentence
                    curr_pred = pred_softmax[es][s]

                    # get the prediction to each possible target sentence in the text
                    row_pred = [0] * max_seq_len
                    for i in range(len(curr_pred)):
                        temp_dist = self.dist_idx_to_dist(i)
                        value = curr_pred[i]
                        if 0  <= temp_dist + s <= max_seq_len-1:
                            row_pred[temp_dist+s] = value

                    attn_matrix.append(row_pred)

                # run MAXIMUM spanning tree
                attn_matrix = np.array(attn_matrix)
                rank_order = get_rank_order(attn_matrix)
                essay_interpretation = run_MST(rank_order, attn_matrix, verdict="max") # --> use the softmax probability as the weight, we run the maximum spanning tree here because higher probability means better

            batch_interpretation.append(essay_interpretation)

        return batch_interpretation


    def _unpack_gold_batch_prediction(self, batch_pred: np.ndarray, seq_len: torch.tensor) -> List:
        """
        Only use predictions without padding

        Args:
            batch_pred (np.ndarray): prediction in batch
            seq_len (torch.Tensor): information about the real length of each essay in the batch

        Returns:
            List
        """
        output = []
        for b in range(len(batch_pred)):
            non_padded_pred = batch_pred[b][:seq_len[b]].tolist()
            non_padded_pred = [self.dist_idx_to_dist(x) for x in non_padded_pred] 
            output.append(non_padded_pred)
        return output
    

    def predict(self, ds: Iterable[Instance], coerce_tree=False) -> np.ndarray:
        """
        Generate prediction result
        coerce_tree = True if we want to make sure that the predictions form a tree (using MST (min or max) algorithm)
        """
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds)) # what if the the valid/test data contain label that does not exist in the training data --> workaround for the vocab
        preds = []
        golds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.extend(self._unpack_model_batch_prediction(batch, coerce_tree=coerce_tree))
                golds.extend(self._unpack_gold_batch_prediction(tonp(batch["rel_dists"]), batch["seq_len"]))

        return preds, golds


class PredictorMTL:
    def __init__(self, 
                model: Model, 
                iterator: DataIterator,
                cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device


    def dist_idx_to_dist(self, idx):
        """
        Args:
            idx (int): index of rel_dist_labels
        """
        return int(self.model.vocab.get_token_from_index(int(idx), namespace="rel_dist_labels"))


    def component_idx_to_label(self, idx):
        """
        Args:
            idx (int): index of rel_dist_labels
        """
        return self.model.vocab.get_token_from_index(int(idx), namespace="component_labels")


    def _unpack_model_batch_prediction(self, batch, coerce_tree=False) -> np.ndarray:
        """
        Interpret prediction result per batch
        coerce_tree = True if we want to make sure that the predictions form a tree (using MST (min or max) algorithm)
        """
        out_dict = self.model(**batch)
        pred_linking_softmax = tonp(out_dict["pred_linking_softmax"])
        pred_node_labelling_softmax = tonp(out_dict["pred_node_labelling_softmax"])

        linking_preds = []
        node_labelling_preds = []
        for es in range(len(pred_linking_softmax)):
            essay_linking = []
            essay_labelling = []
            max_seq_len = batch["seq_len"][es]

            # simple decoding using argmax
            for s in range(max_seq_len): # iterate each sentence in the essay, s is the index of the current sentence
                # perform constrained argmax for linking
                curr_link_softmax = pred_linking_softmax[es][s]
                ranked_pred = [i for i in reversed(sorted(enumerate(curr_link_softmax), key=lambda x:x[1]))]
                for i in range(len(ranked_pred)):
                    tmp_dist = self.dist_idx_to_dist(ranked_pred[i][0])
                    if 0 <= tmp_dist + s <= max_seq_len-1:
                        pred_dist = tmp_dist
                        break

                # argmax for labelling
                curr_label_softmax = pred_node_labelling_softmax[es][s]
                pred_idx = np.argmax(curr_label_softmax)
                pred_label = self.component_idx_to_label(pred_idx)

                # essay-level result
                essay_linking.append(pred_dist)
                essay_labelling.append(pred_label)

            # check if the output is tree
            rep = TreeBuilder(essay_linking)
            if (not rep.is_tree()) and (coerce_tree==True):
                attn_matrix = [] # element [i,j] denotes the probability of sentence i connects to sentence j (j as the target)
                for s in range(max_seq_len): # iterate each sentence in the essay, s is the index of the current sentence
                    curr_pred = pred_linking_softmax[es][s]

                    # get the prediction to each possible target sentence in the text
                    row_pred = [0] * max_seq_len
                    for i in range(len(curr_pred)):
                        temp_dist = self.dist_idx_to_dist(i)
                        value = curr_pred[i]
                        if 0  <= temp_dist + s <= max_seq_len-1:
                            row_pred[temp_dist+s] = value

                    attn_matrix.append(row_pred)

                # run MAXIMUM spanning tree
                attn_matrix = np.array(attn_matrix)
                rank_order = get_rank_order(attn_matrix)
                essay_linking = run_MST(rank_order, attn_matrix, verdict="max") # --> use the softmax probability as the weight, we run the maximum spanning tree here because higher probability means better

            # batch-level result
            linking_preds.append(essay_linking)
            node_labelling_preds.append(essay_labelling)

        return linking_preds, node_labelling_preds


    def _unpack_gold_batch_prediction(self, batch: np.ndarray) -> List:
        """
        Only use predictions without padding

        Args:
            batch (torch.Tensor): prediction in batch

        Returns:
            List
        """
        output_linking = []
        output_node_labelling = []

        batch_rel_dists = tonp(batch["rel_dists"])
        batch_component_labels = tonp(batch["component_labels"])
        seq_len = batch["seq_len"]

        for b in range(len(batch_rel_dists)):
            rel_dists_gold = batch_rel_dists[b][:seq_len[b]].tolist()
            rel_dists_gold = [self.dist_idx_to_dist(x) for x in rel_dists_gold]

            component_labels_gold = batch_component_labels[b][:seq_len[b]].tolist()
            component_labels_gold = [self.component_idx_to_label(x) for x in component_labels_gold]

            output_linking.append(rel_dists_gold)
            output_node_labelling.append(component_labels_gold)
        return output_linking, output_node_labelling
    

    def predict(self, ds: Iterable[Instance], coerce_tree=False) -> np.ndarray:
        """
        Generate prediction result
        coerce_tree = True if we want to make sure that the output forms a tree
        """
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds_linking = []
        preds_node_labelling = []
        golds_linking = []
        golds_node_labelling = []

        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                
                p_linking, p_node_labelling = self._unpack_model_batch_prediction(batch, coerce_tree=coerce_tree)
                g_linking, g_node_labelling = self._unpack_gold_batch_prediction(batch)

                preds_linking.extend(p_linking)
                preds_node_labelling.extend(p_node_labelling)
                golds_linking.extend(g_linking)
                golds_node_labelling.extend(g_node_labelling)

        return preds_linking, golds_linking, preds_node_labelling, golds_node_labelling



class PredictorDep:
    def __init__(self, 
                model: Model, 
                iterator: DataIterator,
                cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device


    def dist_idx_to_dist(self, idx):
        """
        Args:
            idx (int): index of rel_dist_labels
        """
        return int(self.model.vocab.get_token_from_index(int(idx), namespace="rel_dist_labels"))


    def _unpack_model_batch_prediction(self, batch, coerce_tree=False) -> np.ndarray:
        """
        Interpret prediction result per batch
        coerce_tree = True if you want to ensure that the output forms a tree
        """
        out_dict = self.model(**batch)
        pred_matrix = out_dict["pred_matrix"]

        batch_interpretation = []
        for es in range(len(pred_matrix)):
            essay_pred = tonp(pred_matrix[es])
           
            # decoding using simple argmax
            essay_pred = np.argmax(essay_pred, axis=-1)
            dist_interpretation = []
            for i in range(len(essay_pred)):
                dist_interpretation.append(essay_pred[i]-i)

            # check if the output is a tree
            rep = TreeBuilder(dist_interpretation)
            if (not rep.is_tree()) and (coerce_tree==True):
                # run MINIMUM spanning tree
                attn_matrix = tonp(pred_matrix[es])
                attn_matrix = np.array(attn_matrix)
                rank_order = get_rank_order(attn_matrix)
                dist_interpretation = run_MST(rank_order, rank_order, verdict="min") # --> use rank as the weight, "minimum" spanning tree, lower_rank number in rank is better

            # add the decoding result to the batch result
            batch_interpretation.append(dist_interpretation)
        return batch_interpretation


    def _unpack_gold_batch_prediction(self, batch_pred: np.ndarray, seq_len: torch.tensor) -> List:
        """
        Only use predictions without padding

        Args:
            batch_pred (np.ndarray): prediction in batch
            seq_len (torch.Tensor): information about the real length of each essay in the batch

        Returns:
            List
        """
        output = []
        for b in range(len(batch_pred)):
            non_padded_pred = batch_pred[b][:seq_len[b]].tolist()
            non_padded_pred = [self.dist_idx_to_dist(x) for x in non_padded_pred] 
            output.append(non_padded_pred)
        return output
    

    def predict(self, ds: Iterable[Instance], coerce_tree=False) -> np.ndarray:
        """
        Generate prediction result
        coerce_tree = True if we want to make sure that the prediction forms a tree
        """
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds)) # what if the the valid/test data contain label that does not exist in the training data --> workaround for the vocab
        preds = []
        golds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.extend(self._unpack_model_batch_prediction(batch, coerce_tree=coerce_tree))
                golds.extend(self._unpack_gold_batch_prediction(tonp(batch["rel_dists"]), batch["seq_len"]))

        return preds, golds
