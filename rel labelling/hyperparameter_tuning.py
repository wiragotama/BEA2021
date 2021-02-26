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
from common_functions import remove_unwanted_files, list_files_in_dir

from sklearn.metrics import classification_report, confusion_matrix


def param_grid_combinations(args) -> List[Dict]:
    """
    Create a combination of parameter grid
    
    Args:
        args (argparse)
        architecture (str)

    Returns:
        List[Dict]
    """
    fc1_u = ast.literal_eval(args.fc1_u)
    fc2_u = ast.literal_eval(args.fc2_u)
    lstm_u = ast.literal_eval(args.lstm_u)
    dropout_rate = ast.literal_eval(args.dropout_rate)
    batch_size = ast.literal_eval(args.batch_size)

    param_grid = dict(
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        fc1_u=fc1_u,
        fc2_u=fc2_u,
        lstm_u=lstm_u,
    )

    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    return combinations


def concat_hyperparam_config(hyperparam: Dict, config: Config) -> Config:
    """
    A helper function to combine hyperparameter and fixed config into a single configuration

    Args:
        hyperparam (Dict)
        config (Config)

    Returns:
        Config
    """
    output = Config()
    for k, v in hyperparam.items():
        output.set(k, v)
    for k, v in config.items():
        output.set(k, v)
    return output


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
        log_file_only (bool): flag if you want to print this in log file
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
        '-architecture', '--architecture', type=str, help='{"ConcatPairClassifier", "LSTMPairClassifier", "BERTFinetuning", "DistilBERTFinetuning"}', required=True)
    parser.add_argument(
        '-dir', '--dir', type=str, help='dataset directory (CV data)', required=True)
    parser.add_argument(
        '-log', '--log', type=str, help='path to save the running log', required=True)
    parser.add_argument(
        '-fc1_u', '-fc1_u', type=str, help='first hidden layer size to try', required=True)
    parser.add_argument(
        '-lstm_u', '-lstm_u', type=str, help='LSTM unit size (only used when architecture="LSTMPairClassifier")', required=True)
    parser.add_argument(
        '-fc2_u', '-fc2_u', type=str, help='second hidden layer size to try (after concat/LSTM)', required=True)
    parser.add_argument(
        '-dropout_rate', '--dropout_rate', type=str, help='list of dropout_rate to try', required=True)
    parser.add_argument(
        '-batch_size', '--batch_size', type=str, help='list of batch_size to try', required=True)
    parser.add_argument(
        '-epochs', '--epochs', type=str, help='list of epochs to try', required=True)
    args = parser.parse_args()

    # check if architecture is known
    accepted_architectures = set(["ConcatPairClassifier", "LSTMPairClassifier", "BERTFinetuning", "DistilBERTFinetuning"])
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

    # learning_rate & use_extracted_emb
    if args.architecture=="BERTFinetuning" or args.architecture=="DistilBERTFinetuning":
        learning_rate = 2e-5
        use_extracted_emb = False
    else:
        learning_rate = 0.001
        use_extracted_emb = True

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
    config = Config(
        use_percentage=1.0 if args.mode=="real_run" else 0.1, # how much data to use
        emb_dim=768,
        n_labels=4,
        max_seq_len=512, # necessary to limit memory usage
        use_extracted_emb=use_extracted_emb, # set False if BERTFinetuning
    )

    # BERT wordpiece tokenizer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=config.max_seq_len,
        do_lowercase=False,
    )   
    
    # dataset reader
    dataset_reader = PairDatasetReader(
        tokenizer=bert_tokenizer,
        token_indexers={"tokens": token_indexer},
        use_percentage=config.use_percentage,
        use_extracted_emb=config.use_extracted_emb # set False if BERTFinetuning
    )

    # CV training
    header = [k for k, v in hyperparams[0].items()] + ["epoch", "F1-macro", "stdev"]
    logging("\t".join(header), log_file, True)
    for hyperparam in hyperparams:
        # save performance at checkpoint time
        performance_per_epoch = [] 
        for t in range(len(epochs)):
            performance_per_epoch.append([])

        CV_start_time = time.time()

        # idea: alternatively, we can try N times of CV
        CV_count = 0
        for CV_fold in CV_dir:
            train_ds = dataset_reader.read(args.dir+"/"+CV_fold+"/train/")
            valid_ds = dataset_reader.read(args.dir+"/"+CV_fold+"/test/")
            print("%s    # train data: %d    # test data: %d\n" % (CV_fold, len(train_ds), len(valid_ds)))

            # prepare batch for training
            vocab = Vocabulary.from_instances(train_ds) # a must, somehow
            iterator = BasicIterator(batch_size=hyperparam["batch_size"])
            iterator.index_with(vocab) # this is a must for consistency reason

            # batch for prediction
            predict_iterator = BasicIterator(batch_size=32) # not to blow up GPU memory
            predict_iterator.index_with(vocab)

            # model
            model_config = concat_hyperparam_config(hyperparam, config)
            model = get_model(args.architecture, vocab, model_config, torch_device)
            model.to(torch_device)

            # optimizer
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
                predictor = Predictor(model, predict_iterator, cuda_device=cuda_device)
                model_preds, gold_preds = predictor.predict(valid_ds)
                
                # performance report
                print(classification_report(y_true=gold_preds, y_pred=model_preds))

                report = classification_report(y_true=gold_preds, y_pred=model_preds, output_dict=True)
                f1_macro = report['macro avg']['f1-score']
                performance_per_epoch[t].append(f1_macro)

                # conserve memory
                del predictor
                del trainer
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
        for t in range(len(performance_per_epoch)):
            val = [str(v) for k, v in hyperparam.items()] + [str(epochs[t])] + [str(np.average(performance_per_epoch[t]))] + [str(np.std(performance_per_epoch[t]))]
            logging("\t".join(val), log_file)
        print("=============================================")

    # end for hyperparams
    log_file.close()


