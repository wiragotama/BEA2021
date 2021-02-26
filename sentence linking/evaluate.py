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

from sklearn.metrics import classification_report, confusion_matrix

from treebuilder import TreeBuilder
from predict import list_directory
from predict import convert_prediction_to_heuristic_baseline


flatten_list = lambda l: [item for sublist in l for item in sublist]


def list_file(path) -> List[str]:
    """
    List directory existing in path
    """
    return [ os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) ]


def open_prediction_STL(directory: str) -> (List, List):
    """
    Open STL predictions

    Args:
        directory (str)

    Returns:
        {
            List,
            List
        }
    """
    with open(directory+"/link_golds.txt", 'r') as f:
        link_golds = ast.literal_eval(f.readline())
    with open(directory+"/link_preds.txt", 'r') as f:
        link_preds = ast.literal_eval(f.readline())

    return link_golds, link_preds


def open_prediction_MTL(directory: str) -> (List, List, List, List):
    """
    Open MTL predictions

    Args:
        directory (str)

    Returns:
        {
            List,
            List,
            List,
            List
        }
    """
    with open(directory+"/link_golds.txt", 'r') as f:
        link_golds = ast.literal_eval(f.readline())
    with open(directory+"/link_preds.txt", 'r') as f:
        link_preds = ast.literal_eval(f.readline())
    with open(directory+"/node_labelling_golds.txt", 'r') as f:
        labelling_golds = ast.literal_eval(f.readline())
    with open(directory+"/node_labelling_preds.txt", 'r') as f:
        labelling_preds = ast.literal_eval(f.readline())

    return link_golds, link_preds, labelling_golds, labelling_preds


def structured_output_quality(links) -> (List, float, float, float):
    """
    Infer component labels automatically from the structure
    """
    component_labels = []
    tree_ratio = 0
    avg_depth = 0
    avg_leaf_prop = 0
    all_depths = []

    n_essays = len(links)

    for i in range(len(links)):
        rep = TreeBuilder(links[i])
        component_labels.append(rep.auto_component_labels(AC_breakdown=True))

        if rep.is_tree():
            tree_ratio += 1

            # evaluate this only when the output forms a tree
            depth, leaf_prop = rep.tree_depth_and_leaf_proportion()
            avg_depth += depth
            all_depths.append(depth)
            avg_leaf_prop += leaf_prop
    
    return component_labels, float(tree_ratio)/float(n_essays), float(avg_depth)/float(tree_ratio), float(avg_leaf_prop)/float(tree_ratio), all_depths


def f1_per_depth(dist_gold: List, dist_prediction: List, max_depth: int):
    """
    Find at which depth prediction mismatches happen (when the output forms a tree)

    Args:
        dist_gold (List): gold answer per essay
        dist_prediction (List): predicted answer per essay
        max_depth (int): max structure depth in the dataset

    Returns:
        tuple, i.e., (list, list, list)
    """
    gold_all_depth = []
    pred_all_depth = []

    for i in range(len(dist_gold)):
        rep_gold = TreeBuilder(dist_gold[i])
        rep_pred = TreeBuilder(dist_prediction[i])

        if rep_pred.is_tree():
            g_depths = rep_gold.node_depths()
            p_depths = rep_pred.node_depths()

            gold_all_depth.append(g_depths)
            pred_all_depth.append(p_depths)
    
    gold_all_depth_flat = flatten_list(gold_all_depth)
    pred_all_depth_flat = flatten_list(pred_all_depth)

    print("=== Depth prediction performance when output forms a tree ===")
    print(classification_report(y_true = gold_all_depth_flat, y_pred=pred_all_depth_flat, digits=3))
    report = classification_report(y_true = gold_all_depth_flat, y_pred=pred_all_depth_flat, output_dict=True)
    f1s = []
    for i in range(max_depth):
        try:
            f1s.append(report[str(i)]['f1-score'])
        except:
            f1s.append(0.0)

    return f1s


def depth_distribution(gold_all_depth: List, pred_all_depth: List, MAX_DEPTH: int) -> (List, List):
    """
    Calculate the depth distribution of output

    Args:
        gold_all_depth (List): depths in the gold standard
        pred_all_depth (List): depths in prediction output
        MAX_DEPTH (int)

    Returns:
        (list, list)
    """

    gold_depth_distribution = [0.0] * MAX_DEPTH # from depth 0 to MAX_DEPTH-1
    pred_depth_distribution = [0.0] * MAX_DEPTH 
    for i in range(len(gold_all_depth)):
        gold_depth_distribution[gold_all_depth[i]] += 1
    for i in range(len(pred_all_depth)):
        pred_depth_distribution[pred_all_depth[i]] += 1
    for i in range(len(gold_depth_distribution)):
        gold_depth_distribution[i] /= float(len(gold_all_depth))
        pred_depth_distribution[i] /= float(len(pred_all_depth))

    print("=== Depth distribution when output forms a tree ===")
    print("Depth \tGold \tPred")
    for i in range(MAX_DEPTH):
        print("%d \t%.3f \t%.3f" % (i, gold_depth_distribution[i], pred_depth_distribution[i]))
    print()
    
    return gold_depth_distribution, pred_depth_distribution


def get_model_run(model_dir: str):
    """
    Get model run order
    """
    subdir = str(model_dir.split("/")[-1].split("-")[-1])
    return subdir


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Evaluation: linking experiment')
    parser.add_argument(
        '-pred_dir', '--pred_dir', type=str, help='directory saving the prediction results', required=True)
    args = parser.parse_args()

    # model
    model_dirs = list_directory(args.pred_dir)
    print("N models to test %d" % (len(model_dirs)))
    model_dirs.sort()

    # try open one directory to determine STL or MTL
    n_files = len(list_file(model_dirs[0]))
    if n_files == 2:
        setting = "STL"
        print("STL / Dep (linking) evaluation")
    elif n_files == 4:
        setting = "MTL"
        print("MTL evaluation")

    # CONSTANT
    FAR_FORWARD = 14
    FAR_BACKWARD = -19
    MAX_DEPTH = 29

    # performance metrics for linking
    f1_macro_link = []
    f1_weighted_link = []
    acc_link = []
    f1_per_distance = []
    for i in range(FAR_FORWARD - FAR_BACKWARD):
        f1_per_distance.append([])

    # performance metrics for node labelling inferred from the structure
    f1_mc = []
    f1_ac_non_leaf = []
    f1_ac_leaf = []
    f1_non_ac = []
    f1_macro_node_label = []

    # performance metrics structured output quality
    tree_ratio = []
    avg_depth = []
    avg_leaf_prop = []
    f1_depths = []
    gold_depth_distribution_all = []
    pred_depth_distribution_all = []

    # performance metrics for MANUAL node labelling (MTL only)
    f1_mc_mtl = []
    f1_ac_non_leaf_mtl = []
    f1_ac_leaf_mtl = []
    f1_non_ac_mtl = []
    f1_macro_node_label_mtl = []

    f1_mc_alignment = []
    f1_ac_non_leaf_alignment = []
    f1_ac_leaf_alignment = []
    f1_non_ac_alignment = []
    f1_macro_node_label_alignment = []
    acc_mc_alignment = []

    # iterate over models
    for model_dir in model_dirs:
        print("Opening", model_dir)
        if setting=="STL":
            link_golds, link_preds = open_prediction_STL(model_dir)
        elif setting=="MTL":
            link_golds, link_preds, manual_comp_labelling_golds, manual_comp_labelling_preds = open_prediction_MTL(model_dir)

        link_golds_flat = flatten_list(link_golds)
        link_preds_flat = flatten_list(link_preds)

        # counter, to confirm the tendency to connect to dist=-1
        freq = Counter(link_preds_flat).most_common(5)
        print(freq)
        for x in freq:
            print("Counter %d: %.2lf" % (x[0], x[1]/len(link_preds_flat)*100)) 


        # sequence tagging result
        print("=== Sequence tagging evaluation ===")
        print(classification_report(y_true=link_golds_flat, y_pred=link_preds_flat, digits=3))
        report = classification_report(y_true=link_golds_flat, y_pred=link_preds_flat, output_dict=True)
        f1_macro_link.append(report['macro avg']['f1-score'])
        acc_link.append(report['accuracy'])
        f1_weighted_link.append(report['weighted avg']['f1-score'])

        # f1 per target distance
        for i in range(FAR_FORWARD - FAR_BACKWARD):
            try:
                f1_per_distance[i].append(report[str(FAR_BACKWARD+i)]['f1-score'])
            except: # not available
                f1_per_distance[i].append(0.0)

        # structured output quality
        gold_component_labels, gold_tree_ratio, gold_avg_depth, gold_avg_leaf_prop, gold_all_depth = structured_output_quality(link_golds)
        pred_component_labels, pred_tree_ratio, pred_avg_depth, pred_avg_leaf_prop, pred_all_depth = structured_output_quality(link_preds)
        gold_component_labels_flat = flatten_list(gold_component_labels)
        pred_component_labels_flat = flatten_list(pred_component_labels)

        # depth distribution
        gold_depth_distrib, pred_depth_distrib = depth_distribution(gold_all_depth, pred_all_depth, MAX_DEPTH)
        gold_depth_distribution_all.append(gold_depth_distrib)
        pred_depth_distribution_all.append(pred_depth_distrib)

        # automatic argumentative component identification from linking
        print("=== Automatic argumentative component identification (from link structure) ===")
        print(classification_report(y_true=gold_component_labels_flat, y_pred=pred_component_labels_flat, digits=3))
        report = classification_report(y_true=gold_component_labels_flat, y_pred=pred_component_labels_flat, output_dict=True)
        f1_mc.append(report['major claim']['f1-score'])
        f1_ac_non_leaf.append(report['arg comp. (non-leaf)']['f1-score'])
        f1_ac_leaf.append(report['arg comp. (leaf)']['f1-score'])
        f1_non_ac.append(report['non-arg comp.']['f1-score'])
        f1_macro_node_label.append(report['macro avg']['f1-score']) 

        # tree quality of the predicted structure
        tree_ratio.append(pred_tree_ratio)
        avg_depth.append(pred_avg_depth)
        avg_leaf_prop.append(pred_avg_leaf_prop)

        # f1 per depth
        f1_depths.append(f1_per_depth(link_golds, link_preds, MAX_DEPTH))

        if setting=="MTL":
            print("=== manual argumentative component identification ===")
            manual_comp_labelling_golds_flat = flatten_list(manual_comp_labelling_golds)
            manual_comp_labelling_preds_flat = flatten_list(manual_comp_labelling_preds)
            print(classification_report(y_true=manual_comp_labelling_golds_flat, y_pred=manual_comp_labelling_preds_flat, digits=3))
            report = classification_report(y_true=manual_comp_labelling_golds_flat, y_pred=manual_comp_labelling_preds_flat, output_dict=True)
            f1_mc_mtl.append(report['major claim']['f1-score'])
            f1_ac_non_leaf_mtl.append(report['arg comp. (non-leaf)']['f1-score'])
            f1_ac_leaf_mtl.append(report['arg comp. (leaf)']['f1-score'])
            f1_non_ac_mtl.append(report['non-arg comp.']['f1-score'])
            f1_macro_node_label_mtl.append(report['macro avg']['f1-score']) 

            print("=== alignment between automatic vs. manual argumentative component identification ===")
            print(classification_report(y_true=pred_component_labels_flat, y_pred=manual_comp_labelling_preds_flat, digits=3))
            report = classification_report(y_true=pred_component_labels_flat, y_pred=manual_comp_labelling_preds_flat, output_dict=True)
            f1_mc_alignment.append(report['major claim']['f1-score'])
            f1_ac_non_leaf_alignment.append(report['arg comp. (non-leaf)']['f1-score'])
            f1_ac_leaf_alignment.append(report['arg comp. (leaf)']['f1-score'])
            f1_non_ac_alignment.append(report['non-arg comp.']['f1-score'])
            f1_macro_node_label_alignment.append(report['macro avg']['f1-score']) 
            acc_mc_alignment.append(report['accuracy'])


    print("==================================================")
    print("=================                =================")
    print("================= GENERAL RESULT =================")
    print("=================                =================")
    print("==================================================")
    print()

    print("=== Sequence tagging evaluation ===")
    print("Run \tAccuracy \tF1-macro \tF1-weighted")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.3lf \t%.3lf" % (subdir, acc_link[i], f1_macro_link[i], f1_weighted_link[i]))
    print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(acc_link), np.std(acc_link), 
                                                                        np.average(f1_macro_link), np.std(f1_macro_link), 
                                                                        np.average(f1_weighted_link), np.std(f1_weighted_link)
                                                                        ))

    print()
    print("=== F1 performance (avg.) on each target distance ===")
    print("Dist \tF1 (avg) \tstdev")
    for i in range(FAR_FORWARD - FAR_BACKWARD):
        print("%d \t%.3lf \t%.3lf" % (FAR_BACKWARD+i, np.average(f1_per_distance[i]), np.std(f1_per_distance[i]) ))
    
    print()
    print("=== Automatic component identification ===")
    print("F1 MC \tF1 AC (non-leaf) \tF1 AC (leaf) \tF1 Non-AC \tF1-Macro")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.3lf \t%.3lf \t%.3lf \t%.3lf" % (subdir, f1_mc[i], f1_ac_non_leaf[i], f1_ac_leaf[i], f1_non_ac[i], f1_macro_node_label[i]))
    print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(f1_mc), np.std(f1_mc), 
                                                                        np.average(f1_ac_non_leaf), np.std(f1_ac_non_leaf), 
                                                                        np.average(f1_ac_leaf), np.std(f1_ac_leaf),
                                                                        np.average(f1_non_ac), np.std(f1_non_ac),
                                                                        np.average(f1_macro_node_label), np.std(f1_macro_node_label)
                                                                        ))

    print()
    print("=== Structured Output Quality ===")
    print("Run \tTree Ratio \tDepth \tLeaf Proportion")
    for i in range(len(model_dirs)):
        subdir = get_model_run(model_dirs[i])
        print("%s \t%.3lf \t%.1lf \t%.3lf" % (subdir, tree_ratio[i], avg_depth[i], avg_leaf_prop[i]))
    print("Average \t%.3lf (%.3lf) \t%.1lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(tree_ratio), np.std(tree_ratio), 
                                                                    np.average(avg_depth), np.std(avg_depth), 
                                                                    np.average(avg_leaf_prop), np.std(avg_leaf_prop)
                                                                    ))

    print()
    print("=== Depth distribution ===")
    depth_d_avg = np.average(np.array(pred_depth_distribution_all), axis=0)
    depth_d_stdev = np.std(np.array(pred_depth_distribution_all), axis=0)
    print("Depth \tAverage % \tstdev")
    for i in range(len(depth_d_avg)):
        print("%d \t%.3f \t%.3f" % (i, depth_d_avg[i], depth_d_stdev[i]))
    print()


    print()
    print("=== F1 per depth ===")
    depth_performance = np.average(np.array(f1_depths), axis=0)
    depth_stdev = np.std(np.array(f1_depths), axis=0)
    print("Depth \tF1 \tstdev")
    for i in range(len(depth_performance)):
        print("%d \t%.3f \t%.3f" % (i, depth_performance[i], depth_stdev[i]))
    print()


    if setting == "MTL":
        print("==================================================")
        print("=================   MTL          =================")
        print("=================   ADDITIONAL   =================")
        print("=================   PERFORMANCE  =================")
        print("==================================================")
        print()

        print()
        print("=== MANUAL component identification ===")
        print("F1 MC \tF1 AC (non-leaf) \tF1 AC (leaf) \tF1 Non-AC \tF1-Macro")
        for i in range(len(model_dirs)):
            subdir = get_model_run(model_dirs[i])
            print("%s \t%.3lf \t%.3lf \t%.3lf \t%.3lf \t%.3lf" % (subdir, f1_mc_mtl[i], f1_ac_non_leaf_mtl[i], f1_ac_leaf_mtl[i], f1_non_ac_mtl[i], f1_macro_node_label_mtl[i]))
        print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(f1_mc_mtl), np.std(f1_mc_mtl), 
                                                                            np.average(f1_ac_non_leaf_mtl), np.std(f1_ac_non_leaf_mtl), 
                                                                            np.average(f1_ac_leaf_mtl), np.std(f1_ac_leaf_mtl),
                                                                            np.average(f1_non_ac_mtl), np.std(f1_non_ac_mtl),
                                                                            np.average(f1_macro_node_label_mtl), np.std(f1_macro_node_label_mtl)
                                                                            ))

        print()
        print("=== Alignment between automatic vs. manual argumentative component identification ===")
        print("F1 MC \tF1 AC (non-leaf) \tF1 AC (leaf) \tF1 Non-AC \tF1-Macro")
        for i in range(len(model_dirs)):
            subdir = get_model_run(model_dirs[i])
            print("%s \t%.3lf \t%.3lf \t%.3lf \t%.3lf \t%.3lf" % (subdir, f1_mc_alignment[i], f1_ac_non_leaf_alignment[i], f1_ac_leaf_alignment[i], f1_non_ac_alignment[i], f1_macro_node_label_alignment[i]))
        print("Average \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf) \t%.3lf (%.3lf)" % (np.average(f1_mc_alignment), np.std(f1_mc_alignment), 
                                                                            np.average(f1_ac_non_leaf_alignment), np.std(f1_ac_non_leaf_alignment), 
                                                                            np.average(f1_ac_leaf_alignment), np.std(f1_ac_leaf_alignment),
                                                                            np.average(f1_non_ac_alignment), np.std(f1_non_ac_alignment),
                                                                            np.average(f1_macro_node_label_alignment), np.std(f1_macro_node_label_alignment)
                                                                            ))
        print()
        print("Alignment accuracy\t%.3lf (%.3lf)" % (np.average(acc_mc_alignment), np.std(acc_mc_alignment)))


            

