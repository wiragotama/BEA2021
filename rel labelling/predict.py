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
        # to read, use ast.literal_eval()
    f.close()
    print("Output successfully saved to", save_dir+filename)


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Prediction for pairwise classfier model')
    parser.add_argument(
        '-test_dir', '--test_dir', type=str, help='dataset directory (test data)', required=True)
    parser.add_argument(
        '-model_dir', '--model_dir', type=str, help='model directory (containing many models)', required=True)
    parser.add_argument(
        '-pred_dir', '--pred_dir', type=str, help='directory to save prediction results', required=True)
    args = parser.parse_args()

    # device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")")
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    print("Available GPUs "+str(available_gpus))

    # model
    model_dirs = remove_unwanted_files(os.listdir(args.model_dir))
    print("%d models to test" % (len(model_dirs)))
    model_dirs.sort()

    # check model class
    config = get_model_config(args.model_dir+model_dirs[0]+"/")
    model_class = config["class"]
    print("Model architecture:", model_class)

    # Fixed configuration
    config = Config(
        use_percentage=1.0, # how much data to use
        max_seq_len=512, # necessary to limit memory usage
        use_extracted_emb=False if model_class=="BERTFinetuning" or model_class=="DistilBERTFinetuning" else True, # set False if BERTFinetuning
        batch_size=32 # for testing, this does not matter
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
    reader = PairDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_percentage=config.use_percentage,
        use_extracted_emb=config.use_extracted_emb # set False if finetuning
    )

    # performance, for reporting
    keys = ["=", "att", "det", "sup", "macro avg"]
    performances = []
    for i in range(len(keys)):
        performances.append([])


    # load test data (CPU only)
    if torch_device.type=="cpu":
        test_ds = reader.read(args.test_dir)
        print("# Instances in test data", len(test_ds))
        print()

    # iterate over models
    for model_dir in model_dirs:
        # loading test data inside the loop. I am not sure why but this is more memory-friendly for GPU (when run)
        if torch_device.type=="cuda":
            test_ds = reader.read(args.test_dir)
            print("# Instances in test data", len(test_ds))
            print()

        # loading model
        model = load_model(args.model_dir+model_dir+"/", torch_device)
        print("Model architecture:", model.param["class"])
        print("Model params", model.param)
        model.to(torch_device)

        # iterator
        iterator = BasicIterator(batch_size=config.batch_size)
        iterator.index_with(model.vocab) # this is a must for consistency reason

        # predict
        predictor = Predictor(model, iterator, cuda_device=cuda_device)
        test_preds, gold_preds = predictor.predict(test_ds)
        
        # save prediction
        subdir = model_dir.split("/")[-1]
        save_output(args.pred_dir + subdir + "/", "labelling_preds.txt", test_preds.tolist())
        save_output(args.pred_dir + subdir + "/", "labelling_golds.txt", gold_preds.tolist())


        # conserve memory
        del predictor
        del iterator
        del model
        if torch_device.type=="cuda":
            del test_ds
            torch.cuda.empty_cache()
    
