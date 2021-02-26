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
from Nets.BiLSTMparser import *


def load_training_data(args, config):
    """
    A helper function to load training data
    """
    # load training data
    train_ds = dataset_reader.read(args.dir)
    print("# Training Instances", len(train_ds))

    # prepare batch
    vocab = Vocabulary.from_instances(train_ds)
    iterator = BasicIterator(batch_size=config.batch_size)
    iterator.index_with(vocab) # this is a must for consistency reason

    # vocabulary
    if config.setting == "MTL":
        config.set("n_labels", vocab.get_vocab_size("component_labels")) # relation labels
    config.set("n_dists", vocab.get_vocab_size("rel_dist_labels")) # output distances

    return train_ds, vocab, iterator


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Training linking model')
    parser.add_argument(
        '-mode', '--mode', type=str, help='"test_run" or "real_run"', required=True)
    parser.add_argument(
        '-architecture', '--architecture', type=str, help='{"BiLSTMSTL", "BiLSTMMTL", "BertBiLSTMSTL", "BertBiLSTMMTL", "BiaffineBiLSTMdep"}', required=True)
    parser.add_argument(
        '-dir', '--dir', type=str, help='dataset directory (train data)', required=True)
    parser.add_argument(
        '-reduc_dim', '-reduc_dim', type=int, help='dimension reduction layer size', required=True)
    parser.add_argument(
        '-lstm_u', '-lstm_u', type=int, help='LSTM unit size', required=True)
    parser.add_argument(
        '-n_stack', '-n_stack', type=int, help='number of BiLSTM stack', required=True)
    parser.add_argument(
        '-fc_u', '-fc_u', type=int, help='dense layer size (after BiLSTM)', required=True)
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
    parser.add_argument(
        '-mtl_loss', '--mtl_loss', type=str, help='mtl loss {"average", "weighted", "dynamic"}', required=False)
    parser.add_argument(
        '-weight_linking', '--weight_linking', type=float, help='linking weight loss when using weighted mtl_loss', required=False)
    args = parser.parse_args()

    # check if architecture is known
    accepted_architectures = set(["BiLSTMSTL", "BiLSTMMTL", "BertBiLSTMSTL", "BertBiLSTMMTL", "BiaffineBiLSTMdep"])
    if not (args.architecture in accepted_architectures):
        raise Exception("Unknown Architecture!")

    # arguments
    print(str(args)+"\n")

    # cuda device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")")
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    print("Available GPUs "+str(available_gpus))

    # learning rate
    if args.architecture == "BiLSTMSTL" or args.architecture == "BiLSTMMTL" or args.architecture == "BiaffineBiLSTMdep":
        learning_rate = 0.001
        is_finetuning = False
    elif "Bert" in args.architecture:
        learning_rate = 2e-5
        is_finetuning = True

    # epochs
    if args.mode == "test_run":
        print("Max epoch=1 for test run")
        args.epochs = 1
    print("# Epochs =", args.epochs)
    
    # train config
    config = Config(
        use_percentage=1.0, # how much data to use
        emb_dim=768,
        reduc_dim=args.reduc_dim,
        lstm_u=args.lstm_u, 
        n_stack=args.n_stack,
        fc_u=args.fc_u,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        batch_size=args.batch_size, # number of essays 
        max_seq_len=512, # for BERT
        use_extracted_emb=not is_finetuning, # set False if BERTFinetuning
        setting=args.architecture[-3:]
    )
    if config.setting == "MTL":
        config.set("mtl_loss", args.mtl_loss)
        config.set("weight_linking", args.weight_linking)

    # BERT wordpiece tokenizer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=config.max_seq_len,
        do_lowercase=False,
    )   
    def tokenizer(s: str, max_seq_len: int=config.max_seq_len) -> List[str]: # wordpiece tokenizer
        return [Token(x) for x in token_indexer.wordpiece_tokenizer(s)[:max_seq_len]]
    
    # reading dataset
    dataset_reader = SeqDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_extracted_emb=config.use_extracted_emb, # set False if finetuning
        mode=config.setting,
        use_percentage=config.use_percentage,
    )

    # load training data and prepare batching (CPU only)
    if torch_device.type == "cpu": 
        train_ds, vocab, iterator = load_training_data(args, config)

    # train model
    for i in range(args.n_run):
        # loading train data inside the loop. I am not sure why but this is more memory-friendly for GPU (when run)
        if torch_device.type == "cuda":
            train_ds, vocab, iterator = load_training_data(args, config)

        # model
        model = get_model(args.architecture, vocab, config, torch_device)
        model.to(torch_device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # learning rate scheduler
        if is_finetuning: # Bert...
            lr_scheduler = None
        else:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, Params({"type" : "step", "gamma": 0.1, "step_size": args.epochs-10})) # this step size is rather experimental science rather than real science

        train_start_time = time.time()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            iterator=iterator,
            train_dataset=train_ds,
            shuffle=True, # to shuffle the batch
            cuda_device=available_gpus if torch.cuda.device_count() > 1 else cuda_device, # support multiple GPUs
            learning_rate_scheduler=lr_scheduler,
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
        

