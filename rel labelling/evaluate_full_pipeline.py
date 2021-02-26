"""
Author: Jan Wira Gotama Putra
"""
from typing import *
from tqdm import tqdm
import time
import argparse
import ast
import itertools
import os
import numpy as np
from collections import Counter

from datasetreader import * 
from common_functions import remove_unwanted_files
from model_functions import *

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score

from discourseunit import DiscourseUnit, Essay, NO_REL_SYMBOL
from treebuilder import TreeBuilder
from datasetreader import PairDatasetReader
from model_functions import *
from common_functions import *
from evaluate import list_directory

def open_essays(source_files):
    """
    Open essays into list of internal data structure

    Args:
        source_files (List[str]): list of filename to open

    Returns:
        list of Essay
    """
    essays_ann = []
    for source_file in source_files:
        essay = Essay(source_file)
        essays_ann.append(essay)
    return essays_ann


def extract_only_rel_label(rels):
    rel_copy = deepcopy(rels)
    for i in range(len(rel_copy)):
        if rel_copy[i] == "major claim" or rel_copy[i] == "non-AC":
            rel_copy[i] = ""
    return rel_copy


def extract_aci_label(rels):
    rel_copy = deepcopy(rels)
    for i in range(len(rel_copy)):
        if rel_copy[i] == "non-AC":
            pass
        elif rel_copy[i] == "major claim":
            rel_copy[i] = "AC"
        else:
            rel_copy[i] = "AC"
    return rel_copy


def linking_mask(rel):
    """
    Mask the relations for linking
    :param list[str] rel: list of relation combination
    :return: list[str]
    """
    link = deepcopy(rel)
    for i in range(len(link)):
        if link[i] == NO_REL_SYMBOL:
            link[i] = 'n'
        else:
            link[i] = 'y'
    return link


def agreed_relations(rel1, rel2):
    """
    Filter the relations, only on agreed relations 
    :param list[str] rel1: relations in annotation 1
    :param list[str] rel2: relations in annotation 2
    :return: list[str], list[str]
    """
    filter1 = []
    filter2 = []
    n = len(rel1)
    for i in range(n):
        if rel1[i] != NO_REL_SYMBOL and rel2[i] != NO_REL_SYMBOL:
            filter1.append(rel1[i])
            filter2.append(rel2[i])

    return filter1, filter2


def link_labelling_kappa(rel1, rel2):
    r1, r2 = agreed_relations(rel1, rel2)
    return cohen_kappa_score(r1, r2, weights=None)


def flatten(matrix):
    retval = []
    for i in range(len(matrix)):
        retval.extend(matrix[i])
    return retval


def acc_per_essay(pred_essay, test_essay):
    pred_dist, rels = pred_essay.get_rel_distances(mode="original", include_non_arg_units=True)
    pred_rels = extract_only_rel_label(rels)
    pred_aci = extract_aci_label(rels)

    gold_dist, rels = test_essay.get_rel_distances(mode="original", include_non_arg_units=True)
    gold_rels = extract_only_rel_label(rels)
    gold_aci = extract_aci_label(rels)

    point = 0
    for i in range(len(gold_dist)):
        if gold_dist[i] == pred_dist[i] and gold_rels[i] == pred_rels[i] and gold_aci[i] == pred_aci[i]:
            point += 1
    acc = float(point) / float(len(gold_dist))
    return acc


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Test pairwise classfier model')
    parser.add_argument(
        '-train_test_split', '--train_test_split', type=str, help='train_test_split_file)', required=False)
    parser.add_argument(
        '-test_dir', '--test_dir', type=str, help='dataset directory (test data in tsv)', required=True)
    parser.add_argument(
        '-pred_dir', '--pred_dir', type=str, help='model prediction', required=True)
    args = parser.parse_args()

    # get the list of test essays
    if args.train_test_split:
        supposed_code = []
        with open(args.train_test_split) as f:
            for line in f: 
                content = line.strip().split(",")
                essay_code = content[0].strip().split(".")[0][1:]
                verdict = content[1].strip()
                if verdict.lower() == "\"test\"":
                    supposed_code.append(essay_code)
        supposed_code.sort()
        print("# supposed test essays", len(supposed_code))

    # open test essays
    files = list_files_in_dir(args.test_dir)
    files.sort()
    essays = open_essays(files)
    test_essays = []
    for i in range(len(essays)):
        # print(essays[i].essay_code+"\"")
        if args.train_test_split and (essays[i].essay_code in supposed_code):
            test_essays.append(essays[i])
        elif not args.train_test_split:
            test_essays.append(essays[i])
    print("# test essays ", len(test_essays))

    # gold ans
    gold_dist = []
    gold_rels = []
    gold_aci  = []
    gold_pair_links = []
    for test_essay in test_essays:
        dist, rels = test_essay.get_rel_distances(mode="original", include_non_arg_units=True)
        rel_labels = extract_only_rel_label(rels)
        aci = extract_aci_label(rels)
        adj_mat = test_essay.adj_matrix()

        gold_dist.extend(dist)
        gold_rels.extend(rel_labels)
        gold_aci.extend(aci)
        gold_pair_links.extend(flatten(adj_mat))

    # open predictions
    pred_dirs = list_directory(args.pred_dir)
    print("%d models to evaluate" % (len(pred_dirs)))
    pred_dirs.sort()
    
    # test per prediction
    overall_acc = 0
    overall_aci_kappa = 0
    overall_link_kappa = 0
    overall_labelling_kappa = 0

    print("run\tACC\tACI kappa\tLink Kappa\tLabelling Kappa")
    for pred_dir in pred_dirs:
        files = list_files_in_dir(pred_dir)
        preds = open_essays(files)

        # open prediction result
        pred_dist = []
        pred_rels = []
        pred_aci = [] # component labels
        pred_pair_links = []
        idx = 0
        for pred in preds:
            dist, rels = pred.get_rel_distances(mode="original", include_non_arg_units=True)
            rel_labels = extract_only_rel_label(rels)
            aci = extract_aci_label(rels)
            adj_mat = pred.adj_matrix()
            
            pred_dist.extend(dist)
            pred_rels.extend(rel_labels)
            pred_aci.extend(aci)
            pred_pair_links.extend(flatten(adj_mat))

        # calculate accuracy
        point = 0
        for i in range(len(gold_dist)):
            if gold_dist[i] == pred_dist[i] and gold_rels[i] == pred_rels[i] and gold_aci[i] == pred_aci[i]:
                point += 1
        acc = float(point) / float(len(gold_dist))
        overall_acc += acc  

        # calculate IAA
        aci_kappa = cohen_kappa_score(gold_aci, pred_aci, weights=None)
        gold_pair_links_masked = linking_mask(gold_pair_links)
        pred_pair_links_masked = linking_mask(pred_pair_links)
        linking_kappa = cohen_kappa_score(gold_pair_links_masked, pred_pair_links_masked, weights=None)
        labelling_kappa = link_labelling_kappa(gold_pair_links, pred_pair_links)
        overall_aci_kappa += aci_kappa
        overall_link_kappa += linking_kappa
        overall_labelling_kappa += labelling_kappa

        # print output
        print("%s\t%.3lf\t%.2lf\t%.2lf\t%.2lf" % (pred_dir.split("/")[-1].split("-")[-1], acc, aci_kappa, linking_kappa, labelling_kappa))

    print("Avg.\t%.3lf\t%.2lf\t%.2lf\t%.2lf" % ( (overall_acc / float(len(pred_dirs))), 
                                                    (overall_aci_kappa / float(len(pred_dirs))), 
                                                    (overall_link_kappa / float(len(pred_dirs))), 
                                                    (overall_labelling_kappa / float(len(pred_dirs))) 
                                                ))
    