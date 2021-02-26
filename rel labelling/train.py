"""
Author: Jan Wira Gotama Putra
"""
from typing import *
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator

from datasetreader import * 
from model_functions import *


def get_hyperparam(args):
    """
    Create a hyperparameter description
    
    Args:
        args (argparse)
        architecture (str)

    Returns:
        Config
    """
    param_grid = Config(
        fc1_u=args.fc1_u,
        fc2_u=args.fc2_u,
        lstm_u=args.lstm_u,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
    )

    return param_grid


def bert_tokenizer(s: str, max_seq_len: int=512) -> List[str]:
    """
    BERT tokenizer
    """
    return [Token(x) for x in token_indexer.wordpiece_tokenizer(s)[:max_seq_len]]


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Training sentence pair classifier model')
    parser.add_argument(
        '-mode', '--mode', type=str, help='"test_run" or "real_run"', required=True)
    parser.add_argument(
        '-architecture', '--architecture', type=str, help='{"ConcatPairClassifier", "LSTMPairClassifier", "BERTFinetuning", "DistilBERTFinetuning"}', required=True)
    parser.add_argument(
        '-dir', '--dir', type=str, help='dataset directory (train data)', required=True)
    parser.add_argument(
        '-fc1_u', '-fc1_u', type=int, help='first hidden layer size', required=True)
    parser.add_argument(
        '-lstm_u', '-lstm_u', type=int, help='LSTM unit size (only used when architecture="LSTMPairClassifier")', required=True)
    parser.add_argument(
        '-fc2_u', '-fc2_u', type=int, help='second hidden layer size (after concat/LSTM)', required=True)
    parser.add_argument(
        '-dropout_rate', '--dropout_rate', type=float, help='dropout_rate', required=True)
    parser.add_argument(
        '-batch_size', '--batch_size', type=int, help='batch_size', required=True)
    parser.add_argument(
        '-epochs', '--epochs', type=int, help='epochs', required=True)
    parser.add_argument(
        '-n_run', '--n_run', type=int, help='the number of run', required=True)
    parser.add_argument(
        '-model_save_dir', '--model_save_dir', type=str, help='directory to save trained models', required=True)
    args = parser.parse_args()

    # check if architecture is known
    accepted_architectures = set(["ConcatPairClassifier", "LSTMPairClassifier", "BERTFinetuning", "DistilBERTFinetuning"])
    if not (args.architecture in accepted_architectures):
        raise Exception("Unknown Architecture!")
    isFinetuning = True if "BERT" in args.architecture else False
    
    # arguments
    print(str(args)+"\n")

    # cuda device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")")
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    print("Available GPUs "+str(available_gpus))

    # hyperparameter combinations
    hyperparam = get_hyperparam(args)
    print(hyperparam)

    # learning rate
    if args.architecture == "ConcatPairClassifier" or args.architecture == "LSTMPairClassifier":
        learning_rate = 0.001
    elif args.architecture == "BERTFinetuning" or args.architecture == "DistilBERTFinetuning":
        learning_rate = 2e-5

    # epochs
    if args.mode == "test_run":
        print("Max epoch=1 for test run")
        args.epochs = 1
    print("# Epochs =", args.epochs)
    
    # Fixed config
    hyperparam.set("use_percentage", 1.0 if args.mode=="real_run" else 0.1) # how much data to use
    hyperparam.set("emb_dim", 768)
    hyperparam.set("n_labels", 4)
    hyperparam.set("max_seq_len", 512)
    hyperparam.set("use_extracted_emb", not isFinetuning)

    # BERT wordpiece tokenizer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=hyperparam.max_seq_len,
        do_lowercase=False,
    )   
    
    # dataset reader
    dataset_reader = PairDatasetReader(
        tokenizer=bert_tokenizer,
        token_indexers={"tokens": token_indexer},
        use_percentage=hyperparam.use_percentage,
        use_extracted_emb=hyperparam.use_extracted_emb # set False if finetuning
    )

    # load training data and prepare batching (CPU only)
    if torch_device.type == "cpu": 
        # load training data
        train_ds = dataset_reader.read(args.dir)
        print("# Training Instances", len(train_ds))

        # prepare batch
        vocab = Vocabulary.from_instances(train_ds)
        iterator = BasicIterator(batch_size=hyperparam.batch_size)
        iterator.index_with(vocab) # this is a must for consistency reason

    # train model
    for i in range(args.n_run):
        # loading train data inside the loop. I am not sure why but this is more memory-friendly for GPU (when run)
        if torch_device.type == "cuda":
            # load training data
            train_ds = dataset_reader.read(args.dir)
            print("# Training Instances", len(train_ds))

            # prepare batch
            vocab = Vocabulary.from_instances(train_ds)
            iterator = BasicIterator(batch_size=hyperparam.batch_size)
            iterator.index_with(vocab) # this is a must for consistency reason

        # model
        model = get_model(args.architecture, vocab, hyperparam, torch_device)
        model.to(torch_device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # train for until max epochs
        train_start_time = time.time()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_ds,
            shuffle=True, # to shuffle the batch
            cuda_device=available_gpus if torch.cuda.device_count() > 1 else cuda_device, # support multiple GPUs
            num_serialized_models_to_keep=1,
            should_log_parameter_statistics=False,
            num_epochs=args.epochs,
        )
        metrics = trainer.train()
        train_end_time = time.time()
        print('Finished Training, run=%d, epochs=%d, time %.3fmins' % (i+1, args.epochs, (train_end_time-train_start_time)/60.0))

        # save model
        save_model(model, args.model_save_dir+"run-"+str(i+1)+"/")

        # conserve memory
        del trainer
        del optimizer
        del model
        if torch_device.type == "cuda":
            del iterator
            del train_ds # since the train_ds of the previous iteration has been moved to gpu, we need to (force) delete it (because otherweise, it won't get detached but may persists) and then reload again to conserve memory
            torch.cuda.empty_cache()
        

