"""
Author: Jan Wira Gotama Putra

Prediction of sentence links, using greedy decoding or MST
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
from model_functions import *
from Nets.BiLSTMparser import *

from sklearn.metrics import classification_report, confusion_matrix
from predictors import *
import pickle


flatten_list = lambda l: [item for sublist in l for item in sublist]


def list_directory(path) -> List[str]:
    """
    List directory existing in path
    """
    return [ os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]


def save_output(save_dir, filename, content):
    """
    Args:
        save_dir (str): path to directory
        filename (str)
        content (Any)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir+filename, 'w+') as f:
        f.write(str(content))
    f.close()
    print("Output successfully saved to", save_dir+filename)


def convert_prediction_to_heuristic_baseline(test_preds):
    """
    Heuristic baseline prediction
    """
    for i in range(len(test_preds)):
        for j in range(len(test_preds[i])):
            if j == 0:
                test_preds[i][j] = 0
            else:
                test_preds[i][j] = -1


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Prediction: linking experiment')
    parser.add_argument(
        '-test_dir', '--test_dir', type=str, help='dataset directory (test data)', required=True)
    parser.add_argument(
        '-model_dir', '--model_dir', type=str, help='model directory (containing many models)', required=True)
    parser.add_argument(
        '-pred_dir', '--pred_dir', type=str, help='directory to save the prediction result', required=True)
    parser.add_argument(
        '-use_mst', '--use_mst', help='specify whether to coerce the output to form a tree', action='store_true')
    args = parser.parse_args()

    # device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")")
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    print("Available GPUs "+str(available_gpus))

    # model
    model_dirs = list_directory(args.model_dir)
    print("%d models to test" % (len(model_dirs)))
    model_dirs.sort()

    # check model class
    config = get_model_config(model_dirs[0]+"/")
    model_architecture = config.architecture
    print("Model architecture:", model_architecture)
    experimental_setting = model_architecture[-3:]

    # Fixed configuration
    config = Config(
        use_percentage=1.0, # how much data to use
        max_seq_len=512, # necessary to limit memory usage
        use_extracted_emb=False if "Bert" in model_architecture else True, # set False if BERT is part of the architecture
        batch_size=2, # for testing, this does not matter
        setting=model_architecture[-3:] # STL or MTL
    )

    # BERT wordpiece tokenizer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=config.max_seq_len,
        do_lowercase=False,
    )
    def tokenizer(s: str, max_seq_len: int=config.max_seq_len) -> List[str]:
        return [Token(x) for x in token_indexer.wordpiece_tokenizer(s)[:max_seq_len]]

    # dataset reader
    reader = SeqDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_extracted_emb=config.use_extracted_emb, # set False if finetuning
        mode=config.setting,
        use_percentage=config.use_percentage,
    )

    # load test data (CPU only)
    if torch_device.type=="cpu":
        test_ds = reader.read(args.test_dir)
        print("# Test data size", len(test_ds))
        print()

    # iterate over models
    for model_dir in model_dirs:
        # loading test data inside the loop. I am not sure why but this is more memory-friendly for GPU (when run)
        if torch_device.type=="cuda":
            test_ds = reader.read(args.test_dir)
            print("# Test data size", len(test_ds))
            print()

        # loading model
        model = load_model(model_dir+"/", torch_device)
        print("Model architecture:", model.param["architecture"])
        print("Model params", model.param)
        model.to(torch_device)

        # iterator
        iterator = BasicIterator(batch_size=config.batch_size)
        iterator.index_with(model.vocab) # this is a must for consistency reason

        # predict
        if experimental_setting.lower() == "stl":
            predictor = PredictorSTL(model, iterator, cuda_device=cuda_device)
            test_preds, gold_preds = predictor.predict(test_ds, coerce_tree=args.use_mst)

            # heuristic baseline
            # convert_prediction_to_heuristic_baseline(test_preds)
            
            subdir = model_dir.split("/")[-1]
            save_output(args.pred_dir + subdir + "/", "link_preds.txt", test_preds)
            save_output(args.pred_dir + subdir + "/", "link_golds.txt", gold_preds)

            test_preds_flat = flatten_list(test_preds)
            gold_preds_flat = flatten_list(gold_preds)

            # print(classification_report(y_true=gold_preds_flat, y_pred=test_preds_flat, digits=3))

        elif experimental_setting.lower() == "mtl":
            predictor = PredictorMTL(model, iterator, cuda_device=cuda_device)
            link_preds, link_golds, node_labelling_preds, node_labelling_golds = predictor.predict(test_ds, coerce_tree=args.use_mst)

            subdir = model_dir.split("/")[-1]
            save_output(args.pred_dir + subdir + "/", "link_preds.txt", link_preds)
            save_output(args.pred_dir + subdir + "/", "link_golds.txt", link_golds)
            save_output(args.pred_dir + subdir + "/", "node_labelling_preds.txt", node_labelling_preds)
            save_output(args.pred_dir + subdir + "/", "node_labelling_golds.txt", node_labelling_golds)

            link_preds_flat = flatten_list(link_preds)
            link_golds_flat = flatten_list(link_golds)
            node_labelling_preds_flat = flatten_list(node_labelling_preds)
            node_labelling_golds_flat = flatten_list(node_labelling_golds)

            # print(classification_report(y_true=link_golds_flat, y_pred=link_preds_flat, digits=3))
            # print(classification_report(y_true=node_labelling_golds_flat, y_pred=node_labelling_preds_flat, digits=3))

        elif experimental_setting.lower() == "dep":
            predictor = PredictorDep(model, iterator, cuda_device=cuda_device)
            test_preds, gold_preds = predictor.predict(test_ds, coerce_tree=args.use_mst)
            
            subdir = model_dir.split("/")[-1]
            save_output(args.pred_dir + subdir + "/", "link_preds.txt", test_preds)
            save_output(args.pred_dir + subdir + "/", "link_golds.txt", gold_preds)

            test_preds_flat = flatten_list(test_preds)
            gold_preds_flat = flatten_list(gold_preds)



