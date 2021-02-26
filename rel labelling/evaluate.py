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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import DataIterator

from datasetreader import * 
from common_functions import remove_unwanted_files
from model_functions import *

from sklearn.metrics import classification_report, confusion_matrix


def open_prediction(directory: str) -> (List, List):
    """
    Open predictions

    Args:
        directory (str)

    Returns:
        {
            List,
            List
        }
    """
    with open(directory+"/labelling_golds.txt", 'r') as f:
        label_golds = ast.literal_eval(f.readline())
    with open(directory+"/labelling_preds.txt", 'r') as f:
        label_preds = ast.literal_eval(f.readline())

    return label_golds, label_preds


def list_directory(path) -> List[str]:
    """
    List directory existing in path
    """
    return [ os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Test pairwise classfier model')
    parser.add_argument(
        '-pred_dir', '--pred_dir', type=str, help='model prediction', required=True)
    args = parser.parse_args()

    # prediction
    pred_dirs = list_directory(args.pred_dir)
    print("%d models to evaluate" % (len(pred_dirs)))
    pred_dirs.sort()

    # performance, for reporting
    keys = ["=", "att", "det", "sup", "macro avg"]
    performances = []
    for i in range(len(keys)):
        performances.append([])

    # iterate over predictions
    for pred_dir in pred_dirs:
        label_golds, label_preds = open_prediction(pred_dir)
        print(classification_report(y_true=label_golds, y_pred=label_preds))
        print(confusion_matrix(y_true=label_golds, y_pred=label_preds))
        print()
        report = classification_report(y_true=label_golds, y_pred=label_preds, output_dict=True)
        for i in range(len(keys)):
            performances[i].append(report[keys[i]]["f1-score"])
    
    # reporting
    print("Report per run")
    print("\t".join(["run"]+keys))
    for j in range(len(performances[0])): # run
        temp = [str(j)]
        for i in range(len(keys)): # key
            temp.append( "{:.3f}".format(performances[i][j]) )
        print("\t".join(temp))
    print()

    # average performance
    print("Average performance")
    print("\t".join(["average"]+keys))
    temp = []
    for i in range(len(keys)):
        temp.append("{:.3f}".format(np.average(performances[i])) + "(" + "{:.3f}".format(np.std(performances[i])) + ")")
    print("\t".join([""]+temp))
        