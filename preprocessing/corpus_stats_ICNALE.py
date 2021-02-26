"""
Author: Jan Wira Gotama Putra

This script is used to show corpus statistics
"""
import os
import sys
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from discourseunit import DiscourseUnit
from discourseunit import Essay
from discourseunit import NO_REL_SYMBOL
import matplotlib.pyplot as plt
import argparse
from common_functions import list_files_in_dir, open_essays, print_stats
from treebuilder import TreeBuilder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Corpus statistics')
    parser.add_argument(
        '-dir', '--dir', type=str, help='relative directory of corpus (html files)', required=True)
    parser.add_argument(
        '-original_order', '--original_order', help='specify whether to use sentence position original order', action='store_true')
    args = parser.parse_args()

    directory = args.dir

    files = list_files_in_dir(directory)
    files.sort()
    essays = open_essays(files)

    # common stats
    n_sentences = [] # the number of sentences per essay
    n_tokens = []
    n_ACs = [] # the number of ACs per essay
    n_non_ACs = [] # the number of non-ACs per essay
    n_depth = [] # depth of the structure
    leaf_prop = [] # the proportion of leaf nodes

    # relations related stats
    n_sup, n_det, n_att, n_res = [], [], [], []
    corpus_dist = []
    backward_per_essay = []
    forward_per_essay = []

    max_token_per_sentence = 0

    # text repair statistics
    mc_repair = 0
    non_mc_repair = 0

    # non_ac_samples
    non_ac_samples = []

    for essay in essays:
        n_sentences.append(len(essay.units))
        n_tokens.append(essay.n_tokens())
        n_ACs.append(essay.n_ACs())
        n_non_ACs.append(essay.n_non_ACs())
        n_depth.append(essay.depth(essay.adj_matrix()))

        n_sup.append(essay.n_rel("sup"))
        n_det.append(essay.n_rel("det"))
        n_att.append(essay.n_rel("att"))
        n_res.append(essay.n_rel("="))

        # about distance
        if args.original_order: # directions without non-ACs
            directions = essay.get_rel_distances("original")[0]  
        else:
            directions = essay.get_rel_distances("reordering")[0] # reordering
        dir_non_zero = np.array([x for x in directions if x != 0]) # we do not care self-loop (in major claim)
        backward_per_essay.append(np.count_nonzero(dir_non_zero <= -1))
        forward_per_essay.append(np.count_nonzero(dir_non_zero >= 1))
        corpus_dist.extend(directions)

        mc_r, non_mc_rep = essay.text_repair_stats()
        mc_repair += mc_r
        non_mc_repair += non_mc_rep

        # to get non_AC samples
        non_ac_samples.extend(essay.get_non_ACS("original", False))

        # about tree structure
        if args.original_order:
            directions_with_non_AC = essay.get_rel_distances("original", include_non_arg_units=True)[0] 
        else:
            directions_with_non_AC = essay.get_rel_distances("reordering", include_non_arg_units=True)[0] # reordering
        try:
            rep = TreeBuilder(directions_with_non_AC) # distances between sentences
        except:
            print("Distance error", essay.essay_code)
        depth, leaf_ratio = rep.tree_depth_and_leaf_proportion()
        leaf_prop.append(leaf_ratio)


    print("> Corpus", directory)
    print("> items", len(essays))
    print("> Common Stats")
    print("  \t\t\t \tsum \tmax \tmin \tavg \tstdev")
    print_stats("# Sentences\t", n_sentences)
    print_stats("# Tokens\t", n_tokens)
    print_stats("# Arg. components", n_ACs)
    print_stats("# Non-arg. comp.", n_non_ACs)
    
    print("> Relations")
    print_stats("# Support\t", n_sup)
    print_stats("# Detail\t", n_det)
    print_stats("# Attack\t", n_att)
    print_stats("# Restatement", n_res)
    print_stats("# Structure Depth", n_depth, print_total=False)
    print_stats("# Leaf Ratio", leaf_prop, print_total=False)
    
    # corpus_dist = np.array([x for x in corpus_dist if x != 0]) # we do not care self-loop (in major claim)
    corpus_dist = np.array(corpus_dist)
    far_backward = max(-30, min(corpus_dist))
    far_forward = min(15, max(corpus_dist))

    print("\n> Relation direction")
    print_stats("# Backward\t", backward_per_essay)
    print_stats("# Forward\t", forward_per_essay)
    print("    # <= %d\t\t\t%d (%.2f%%)" % ( far_backward, np.count_nonzero(corpus_dist <= far_backward), np.count_nonzero(corpus_dist <= far_backward)/len(corpus_dist)*100 ))
    for i in range(far_backward+1, far_forward):
        if i!=0:
            print("    # == %d\t\t\t%d (%.2f%%)" % ( i, np.count_nonzero(corpus_dist == i), np.count_nonzero(corpus_dist == i)/len(corpus_dist)*100 ))
    print("    # >= %d\t\t\t%d (%.2f%%)" % ( far_forward, np.count_nonzero(corpus_dist >= far_forward), np.count_nonzero(corpus_dist >= far_forward)/len(corpus_dist)*100 ))

    # statistics of sentence repair + major claim repair
    print("\n> Text repair statistics")
    print("MC", mc_repair)
    print("non-MC", non_mc_repair)
