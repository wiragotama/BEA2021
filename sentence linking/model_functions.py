"""
Author: Jan Wira Gotama Putra
"""
from typing import *
from tqdm import tqdm
import time
import numpy as np
import os
import json
import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator
from allennlp.common.util import namespace_match
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.common import Params

from Nets.BiLSTMparser import *
from Nets.BertBiLSTM import *
from Nets.BiaffineBiLSTMdep import *

from sklearn.metrics import classification_report, confusion_matrix
from datasetreader import *


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


def save_model(model: Model , save_dir: str) -> None:
    """
    Save model to directory
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # model weight
    with open(save_dir+"model.th", 'wb') as f:
        torch.save(model.state_dict(), f)
    # vocabulary
    model.vocab.save_to_files(save_dir+"vocabulary/")
    # model config
    with open(save_dir+"config", 'w+') as f:
        json.dump(model.param, f)
    print("Model successfully saved to", save_dir)


def load_vocab_from_directory(directory: str, padding_token: str="[PAD]", oov_token: str="[UNK]") -> Vocabulary:
    """
    Load pre-trained vocabulary form a directory (since the original method does not work --> OOV problem)
    
    Args:
        directory (str)
        padding_token (str): default OOV token symbol ("[PAD]" our case, since we are using BERT)
        oov_token (str): default OOV token symbol ("[UNK]" our case, since we are using BERT)

    Returns:
        Vocabulary
    """
    NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'

    print("Loading token dictionary from", directory)
    with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'r', 'utf-8') as namespace_file:
        non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]

    vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)

    # Check every file in the directory.
    for namespace_filename in os.listdir(directory):
        if namespace_filename == NAMESPACE_PADDING_FILE:
            continue
        if namespace_filename.startswith("."):
            continue
        namespace = namespace_filename.replace('.txt', '')
        if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
            is_padded = False
        else:
            is_padded = True
        filename = os.path.join(directory, namespace_filename)
        vocab.set_from_file(filename, is_padded, oov_token=oov_token, namespace=namespace)
        vocab._padding_token = padding_token

    return vocab


def load_model(save_dir: str, torch_device: torch.device, padding_token: str="[PAD]", oov_token: str="[UNK]") -> Model:
    """
    Load a model from directory

    Args:
        save_dir (str)
        torch_device (torch.device)
        padding_token (str): symbol for padding token
        ovv_token (str): symbol for oov token
    """
    print("Loading model from", save_dir)

    # load existing vocabulary, this is actually very dirty since we need to set padding and oov token manually, but this is the current workaround
    vocab = load_vocab_from_directory(directory=save_dir+"vocabulary/", padding_token=padding_token, oov_token=oov_token) 

    # config
    config = get_model_config(save_dir)

    # create model instance, and then load weight
    model = get_model(config.architecture, vocab, config, torch_device)
    model.load_state_dict(torch.load(save_dir+"model.th", map_location=torch_device))
    return model


def get_model_config(save_dir: str) -> Dict:
    """
    Load a model config

    Returns:
        Dict
    """
    # config
    with open(save_dir+"config") as json_file:
        config_json = json.load(json_file)
    config = Config()
    for k, v in config_json.items():
        config.set(k, v)
    return config


def get_model(architecture:str, vocab: Vocabulary, hyperparam: Config, torch_device: torch.device) -> Model:
    """
    Get a model

    Args:
        architecture (str): architecture type
        vocab (Vocabulary)
        hyperparam (Dict): hyperparam to try for this model
        torch_device (torch.device)

    Returns:
        Model
    """
    if architecture=="BiLSTMSTL":
        print("BiLSTM (STL) with number of dist:", hyperparam.n_dists)
        return BiLSTMSTL(
            vocab=vocab,
            emb_dim=hyperparam.emb_dim, 
            reduc_dim=hyperparam.reduc_dim,
            lstm_u=hyperparam.lstm_u,
            n_stack=hyperparam.n_stack,
            fc_u=hyperparam.fc_u,
            n_dists=hyperparam.n_dists,
            dropout_rate=hyperparam.dropout_rate,
            torch_device=torch_device
            )
    elif architecture=="BiLSTMMTL":
        print("BiLSTM (MTL) with number of dist:", hyperparam.n_dists)
        print("BiLSTM (MTL) with number of component label:", hyperparam.n_labels)
        return BiLSTMMTL(
            vocab=vocab,
            emb_dim=hyperparam.emb_dim, 
            reduc_dim=hyperparam.reduc_dim,
            lstm_u=hyperparam.lstm_u,
            n_stack=hyperparam.n_stack,
            fc_u=hyperparam.fc_u,
            n_dists=hyperparam.n_dists,
            n_labels=hyperparam.n_labels,
            dropout_rate=hyperparam.dropout_rate,
            torch_device=torch_device,
            mtl_loss=hyperparam.mtl_loss,
            weight_linking=hyperparam.weight_linking
            )
    elif architecture=="BertBiLSTMSTL":
        print("BERT-BiLSTM (STL) with number of dist:", hyperparam.n_dists)
        return BertBiLSTMSTL(
            vocab=vocab,
            emb_dim=hyperparam.emb_dim, 
            lstm_u=hyperparam.lstm_u,
            n_stack=hyperparam.n_stack,
            fc_u=hyperparam.fc_u,
            n_dists=hyperparam.n_dists,
            dropout_rate=hyperparam.dropout_rate,
            torch_device=torch_device
            )
    elif architecture=="BertBiLSTMMTL":
        print("BERT-BiLSTM (MTL) with number of dist:", hyperparam.n_dists)
        print("BERT-BiLSTM (MTL) with number of component label:", hyperparam.n_labels)
        return BertBiLSTMMTL(
            vocab=vocab,
            emb_dim=hyperparam.emb_dim, 
            lstm_u=hyperparam.lstm_u,
            n_stack=hyperparam.n_stack,
            fc_u=hyperparam.fc_u,
            n_dists=hyperparam.n_dists,
            n_labels=hyperparam.n_labels,
            dropout_rate=hyperparam.dropout_rate,
            torch_device=torch_device,
            mtl_loss=hyperparam.mtl_loss,
            weight_linking=hyperparam.weight_linking
            )
    elif architecture=="BiaffineBiLSTMdep":
        print("BiaffineBiLSTM for dependency parsing with number of dist:", hyperparam.n_dists)
        return BiaffineBiLSTMdep(
            vocab=vocab,
            emb_dim=hyperparam.emb_dim, 
            reduc_dim=hyperparam.reduc_dim,
            lstm_u=hyperparam.lstm_u,
            n_stack=hyperparam.n_stack,
            fc_u=hyperparam.fc_u,
            n_dists=hyperparam.n_dists,
            dropout_rate=hyperparam.dropout_rate,
            torch_device=torch_device
            )
    else:
        return None


if __name__ == "__main__":
    working_dir = "../data/ICNALE-BERT/original/linking/train/"
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_class = "BiaffineBiLSTMdep"
    is_finetuning = True if "Bert" in model_class else False # set False if not Finetuning
    setting = model_class[-3:]
    model_save_dir = "test_run/" + setting + "/"

    config = Config(
        use_percentage=1.0, # how much data to use
        emb_dim=768,
        reduc_dim=512,
        lstm_u=256, 
        n_stack=3,
        fc_u=256,
        dropout_rate=0.5,
        epochs=70,
        batch_size=8, # number of essays 
        max_seq_len=512, # for BERT
        use_extracted_emb=not is_finetuning, # set False if BERTFinetuning
        setting=setting,
        mtl_loss="dynamic",
        weight_linking=0.7
    )
    print(config)
    print("device", torch_device)

    # BERT wordpiece tokenizer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=config.max_seq_len,
        do_lowercase=False,
    )
    def tokenizer(s: str, max_seq_len: int=config.max_seq_len) -> List[str]:
        return [Token(x) for x in token_indexer.wordpiece_tokenizer(s)[:max_seq_len]]
    if "Bert" in model_class:
        print("\n!!! TRIAL: set tokenizer=None for Sentence-BERT BiLSTM !!!\n")
        tokenizer = None

    # reading dataset
    reader = SeqDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_extracted_emb=config.use_extracted_emb, # set False if finetuning
        mode=config.setting,
        use_percentage=config.use_percentage,
    )

    train_ds = reader.read(working_dir)
    print("Training data size", len(train_ds))

    # vocabulary
    vocab = Vocabulary.from_instances(train_ds) # the tutorial does not work (https://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/)
    if config.setting == "MTL":
        config.set("n_labels", vocab.get_vocab_size("component_labels")) # relation labels
    config.set("n_dists", vocab.get_vocab_size("rel_dist_labels")) # output distances

    # prepare batch
    iterator = BasicIterator(batch_size=config.batch_size)
    iterator.index_with(vocab) # this is a must for consistency reason

    # model
    model = get_model(model_class, vocab, config, torch_device)
    model.to(torch_device)
    
    # training
    optimizer = optim.Adam(model.parameters(), lr=2e-5 if is_finetuning else 0.001) # lr = 2e-5 for BERT, 0.001 if not
    train_start_time = time.time()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        shuffle=True, # to shuffle the batch
        cuda_device=cuda_device,
        should_log_parameter_statistics=False,
        learning_rate_scheduler=None,
        num_epochs=config.epochs,
    )
    metrics = trainer.train()
    train_end_time = time.time()
    print('Finished Training, time %.3fmins' % ((train_end_time-train_start_time)/60.0))

    # save model
    save_model(model, model_save_dir + "run-test/")
    model2 = load_model(model_save_dir + "run-test/", torch_device)





