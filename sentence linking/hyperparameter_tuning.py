"""
Author: Jan Wira Gotama Putra
"""
from typing import *
from tqdm import tqdm
import time
import numpy as np
import argparse
import ast
import itertools
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator

from datasetreader import * 
from model_functions import *
from Nets.BiLSTMparser import *
from predictors import * 
from predict import flatten_list

from common_functions import remove_unwanted_files, list_files_in_dir

from sklearn.metrics import classification_report, confusion_matrix


def param_grid_combinations(args):
    """
    Create a combination of parameter grid
    
    Args:
        args (argparse)

    Returns:
        list[dict]
    """
    reduc_dim = ast.literal_eval(args.reduc_dim)
    lstm_u = ast.literal_eval(args.lstm_u)
    n_stack = ast.literal_eval(args.n_stack)
    fc_u = ast.literal_eval(args.fc_u)
    dropout_rate = ast.literal_eval(args.dropout_rate)
    batch_size = ast.literal_eval(args.batch_size) 

    param_grid = dict(
        reduc_dim=reduc_dim,
        lstm_u=lstm_u,
        n_stack=n_stack,
        fc_u=fc_u,
        dropout_rate=dropout_rate,
        batch_size=batch_size
    )

    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    return combinations


def bert_tokenizer(s: str, max_seq_len: int=512) -> List[str]:
    """
    BERT tokenizer
    """
    return [Token(x) for x in token_indexer.wordpiece_tokenizer(s)[:max_seq_len]]


def logging(s: str, log_file, log_file_only: bool=False):
    """
    A helper function

    Args:
        s (str): the message you want to log
        log_file: log file object
        log_file_only (bool): flag if you want to print this in log file only
    """
    if not log_file_only:
        print(s)
    log_file.write(s+"\n")
    log_file.flush()


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning using cross-validation')
    parser.add_argument(
        '-mode', '--mode', type=str, help='"test_run" or "real_run"', required=True)
    parser.add_argument(
        '-architecture', '--architecture', type=str, help='{"BiLSTMSTL", "BiLSTMMTL", "BertBiLSTMSTL", "BertBiLSTMMTL", "BiaffineBiLSTMdep"}', required=True)
    parser.add_argument(
        '-dir', '--dir', type=str, help='dataset directory (CV data)', required=True)
    parser.add_argument(
        '-reduc_dim', '-reduc_dim', type=str, help='list of dimension reduction layer size', required=True)
    parser.add_argument(
        '-lstm_u', '-lstm_u', type=str, help='list of LSTM unit size', required=True)
    parser.add_argument(
        '-n_stack', '-n_stack', type=str, help='list of number of BiLSTM stack', required=True)
    parser.add_argument(
        '-fc_u', '-fc_u', type=str, help='list of dense layer size (after BiLSTM)', required=True)
    parser.add_argument(
        '-dropout_rate', '--dropout_rate', type=str, help='list of dropout_rate', required=True)
    parser.add_argument(
        '-batch_size', '--batch_size', type=str, help='list of batch_size', required=True)
    parser.add_argument(
        '-epochs', '--epochs', type=str, help='list of epochs', required=True)
    parser.add_argument(
        '-mtl_loss', '--mtl_loss', type=str, help='mtl loss {"average", "weighted", "dynamic"}', required=False)
    parser.add_argument(
        '-weight_linking', '--weight_linking', type=float, help='linking weight loss when using weighted mtl_loss', required=False)
    parser.add_argument(
        '-log', '--log', type=str, help='path to save the running log', required=True)
    args = parser.parse_args()

    # check if architecture is known
    accepted_architectures = set(["BiLSTMSTL", "BiLSTMMTL", "BertBiLSTMSTL", "BertBiLSTMMTL", "BiaffineBiLSTMdep"])
    if not (args.architecture in accepted_architectures):
        raise Exception("Unknown Architecture!")

    # safety measure
    if os.path.exists(args.log):
        raise Exception("Existing log file with the same name exists! Use the different name!")

    # log file
    log_file = open(args.log,"w+")
    csv_separator = "\t"

    # arguments
    logging(str(args)+"\n", log_file)

    # cuda device
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")", log_file)
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    logging("Available GPUs "+str(available_gpus), log_file)

    # hyperparameter combinations
    hyperparams = param_grid_combinations(args)
    logging("Param combinations = "+str(len(hyperparams)), log_file)

    # experimental setting
    logging("Experimental setting = "+str(args.architecture[-3:]), log_file)
    logging("MTL loss = " + str(args.mtl_loss), log_file)
    logging("MTL weight_linking = " + str(args.weight_linking), log_file)

    # learning rate
    if args.architecture=="BiLSTMSTL" or args.architecture=="BiLSTMMTL" or args.architecture == "BiaffineBiLSTMdep":
        learning_rate = 0.001
        is_finetuning = False
        tokenizer=bert_tokenizer
    elif "Bert" in args.architecture:
        learning_rate = 2e-5
        is_finetuning = True
        print("\n!!! TRIAL: set tokenizer=None for Sentence-BERT !!!\n")
        tokenizer=None

    # epochs
    if args.mode == "real_run":
        epochs = ast.literal_eval(args.epochs)
        epochs.sort()
    else: # test run
        print("Max epoch=1 for test run")
        epochs = [1]
    logging("# Epochs to try (checkpoints) = "+str(len(epochs)), log_file)
    
    # CV directory
    CV_dir = remove_unwanted_files(os.listdir(args.dir))
    CV_dir.sort()
    logging("Experiment of "+str(len(CV_dir))+"-Cross Validation\n", log_file)

    # Fixed config
    fixedconfig = Config(
        use_percentage=1.0, # how much data to use
        emb_dim=768,
        max_seq_len=256, # necessary to limit memory usage
        use_extracted_emb=not is_finetuning, # set False if BERTFinetuning
        setting=args.architecture[-3:]
    )

    # BERT wordpiece tokenizer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=fixedconfig.max_seq_len,
        do_lowercase=False,
    )   
    
    # dataset reader
    dataset_reader = SeqDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_percentage=fixedconfig.use_percentage,
        mode=fixedconfig.setting,
        use_extracted_emb=fixedconfig.use_extracted_emb # set False if BERTFinetuning
    )

    # CV training
    header = [k for k, v in hyperparams[0].items()] + ["epoch", "F1-macro (linking)", "Accuracy (linking)"]
    if fixedconfig.setting == "MTL":
        header = header + ["F1-macro (node labelling)", "Accuracy (node labelling)"]
    logging("\t".join(header), log_file, True)
    for hyperparam in hyperparams:
        # save performance at checkpoint time
        f1_linking_per_ep = [] 
        acc_linking_per_ep = []
        if fixedconfig.setting == "MTL":
            f1_node_label = []
            acc_node_label = []

        for t in range(len(epochs)):
            f1_linking_per_ep.append([])
            acc_linking_per_ep.append([])
            if fixedconfig.setting == "MTL":
                f1_node_label.append([])
                acc_node_label.append([])

        # Cross validation
        CV_start_time = time.time()
        CV_count = 0
        for CV_fold in CV_dir:
            train_ds = dataset_reader.read(args.dir+"/"+CV_fold+"/train/")
            valid_ds = dataset_reader.read(args.dir+"/"+CV_fold+"/test/")
            print("%s    # train data: %d    # test data: %d\n" % (CV_fold, len(train_ds), len(valid_ds)))

            # prepare batch for training
            combined_dataset = train_ds + valid_ds
            vocab = Vocabulary.from_instances(combined_dataset) # a workaround so the model can predict unseen label in the training data
            del combined_dataset # saving memory
            iterator = BasicIterator(batch_size=hyperparam["batch_size"])
            iterator.index_with(vocab) # this is a must for consistency reason

            # batch for prediction
            predict_iterator = BasicIterator(batch_size=1) # not to blow up GPU memory
            predict_iterator.index_with(vocab)

            # current hyperparameter + setting
            current_model_hyperparam = Config()
            for k, v in hyperparam.items(): current_model_hyperparam.set(k, v) 
            current_model_hyperparam.set("emb_dim", fixedconfig.emb_dim)
            # n output labels
            if fixedconfig.setting == "MTL":
                current_model_hyperparam.set("n_labels", vocab.get_vocab_size("component_labels")) # relation labels
                current_model_hyperparam.set("mtl_loss", args.mtl_loss)
                current_model_hyperparam.set("weight_linking", args.weight_linking)
            current_model_hyperparam.set("n_dists", vocab.get_vocab_size("rel_dist_labels")) # output distances

            # model
            model = get_model(args.architecture, vocab, current_model_hyperparam, torch_device)
            model.to(torch_device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 2e-5 for BERT, 0.001 for others

            # train for until max epochs, with checkpoint-ing
            train_start_time = time.time()
            for t in range(len(epochs)):
                if t > 0:
                    train_epoch = epochs[t] - epochs[t-1] 
                else:
                    train_epoch = epochs[t]
                
                trainer = Trainer(
                    model=model,
                    optimizer=optimizer,
                    iterator=iterator,
                    train_dataset=train_ds,
                    shuffle=True, # to shuffle the batch
                    cuda_device=available_gpus if torch.cuda.device_count() > 1 else cuda_device,
                    num_serialized_models_to_keep=1,
                    should_log_parameter_statistics=False,
                    num_epochs=train_epoch,
                )
                metrics = trainer.train()
                train_end_time = time.time()
                print('Finished Training, %d epochs, time %.3fmins' % (epochs[t], (train_end_time-train_start_time)/60.0))

                # make prediction on validation data
                if fixedconfig.setting == "STL":
                    predictor = PredictorSTL(model, iterator, cuda_device=cuda_device)
                    link_preds, link_golds = predictor.predict(valid_ds)
                elif fixedconfig.setting == "MTL":
                    predictor = PredictorMTL(model, iterator, cuda_device=cuda_device)
                    link_preds, link_golds, labelling_preds, labelling_golds = predictor.predict(valid_ds)
                elif fixedconfig.setting == "dep":
                    predictor = PredictorDep(model, iterator, cuda_device=cuda_device)
                    link_preds, link_golds = predictor.predict(valid_ds)
                
                link_preds_flat = flatten_list(link_preds)
                gold_preds_flat = flatten_list(link_golds)

                # performance report linking
                print(classification_report(y_true=gold_preds_flat, y_pred=link_preds_flat, digits=3))
                linking_report = classification_report(y_true=gold_preds_flat, y_pred=link_preds_flat, output_dict=True)

                # performance report node labelling
                if fixedconfig.setting == "MTL":
                    labelling_preds_flat = flatten_list(labelling_preds)
                    labelling_golds_flat = flatten_list(labelling_golds)
                    print(classification_report(y_true=labelling_golds_flat, y_pred=labelling_preds_flat, digits=3))
                    node_label_report = classification_report(y_true=labelling_golds_flat, y_pred=labelling_preds_flat, output_dict=True)

                # get the linking score
                f1_linking_per_ep[t].append(linking_report['macro avg']['f1-score'])
                acc_linking_per_ep[t].append(linking_report['accuracy'])

                # labelling report for MTL
                if fixedconfig.setting == "MTL":
                    f1_node_label[t].append(node_label_report['macro avg']['f1-score'])
                    acc_node_label[t].append(node_label_report['accuracy'])

                # conserve memory
                del predictor
                del trainer
                del link_preds_flat
                del gold_preds_flat
                del link_preds
                del link_golds
                if torch_device.type == "cuda":
                    torch.cuda.empty_cache()

            # conserve memory
            del iterator
            del predict_iterator
            del optimizer
            del model
            del vocab
            del train_ds
            del valid_ds
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            # end for epochs
        # end for CV

        # average performance across folds
        CV_end_time = time.time()
        print("=============================================")
        print('Time elapsed one K-fold-CV %.3fmins' % ((CV_start_time-CV_end_time)/60.0))
        for t in range(len(f1_linking_per_ep)):
            val = [str(v) for k, v in hyperparam.items()] + [str(epochs[t])] + [str(np.average(f1_linking_per_ep[t]))] + [str(np.average(acc_linking_per_ep[t]))]
            if fixedconfig.setting == "MTL":
                val = val + [str(np.average(f1_node_label[t]))] + [str(np.average(acc_node_label[t]))]
            logging("\t".join(val), log_file)
        print("=============================================")

    # end for hyperparams
    log_file.close()


