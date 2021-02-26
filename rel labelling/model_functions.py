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
from allennlp.nn.util import get_text_field_mask

from sklearn.metrics import classification_report, confusion_matrix
from datasetreader import * 
from FFnet import * 
from BERT import *


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
    with open(save_dir+"config") as json_file:
        configuration = json.load(json_file)

    # make model instance, and then load weight
    config = Config()
    for k, v in configuration.items():
        config.set(k, v)

    model = get_model(configuration["class"], vocab, config, torch_device)
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
        config = json.load(json_file)
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
    if architecture=="ConcatPairClassifier":
        return ConcatPairClassifier(
            vocab=vocab,
            emb_dim=hyperparam.emb_dim, 
            fc1_u=hyperparam.fc1_u,
            fc2_u=hyperparam.fc2_u,
            dropout_rate=hyperparam.dropout_rate,
            n_labels=hyperparam.n_labels 
        )
        return model
    elif architecture=="LSTMPairClassifier":
        return LSTMPairClassifier(
            vocab=vocab,
            emb_dim=hyperparam.emb_dim, 
            fc1_u=hyperparam.fc1_u,
            fc2_u=hyperparam.fc2_u,
            lstm_u=hyperparam.lstm_u,
            dropout_rate=hyperparam.dropout_rate,
            n_labels=hyperparam.n_labels 
        )
    elif architecture=="BERTFinetuning":
        return BERTFinetuning(
            vocab=vocab,
            n_labels=hyperparam.n_labels,
            torch_device=torch_device
            )
    elif architecture=="DistilBERTFinetuning":
        return DistilBERTFinetuning(
            vocab=vocab,
            n_labels=hyperparam.n_labels,
            torch_device=torch_device
            )
    else:
        return None


def tonp(tsr): return tsr.detach().cpu().numpy() # for seamless interaction between gpu and cpu


class Predictor:
    def __init__(self, 
                model: Model, 
                iterator: DataIterator,
                cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device


    def _extract_data(self, batch) -> np.ndarray:
        """
        Interpret prediction result per batch
        """
        out_dict = self.model(**batch)
        label_int = np.argmax(tonp(out_dict["class_log_softmax"]), axis=-1)
        return label_int


    def _interpret_prediction(self, preds) -> np.ndarray:
        """
        Interpret prediction result

        Args:
            preds (array-like object)
        """
        label_str = []
        for x in preds:
            label_str.append(self.model.vocab.get_token_from_index(int(x), namespace="labels"))
        return np.array(label_str)
    

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        """
        Generate prediction result
        """
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        golds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
                golds.append(tonp(batch["label"]))

        return self._interpret_prediction(np.concatenate(preds, axis=0)), self._interpret_prediction(np.concatenate(golds, axis=0))


if __name__ == "__main__":
    working_dir = "../data/ICNALE-SBERT/original/pairwise_link_labelling/train/"
    model_architecture = "ConcatPairClassifier"
    use_extracted_emb = False if "BERT" in model_architecture else True 

    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", torch_device)

    config = Config(
        use_percentage=1.0, # how much data to use
        emb_dim=768,
        fc1_u=256,
        fc2_u=256,
        lstm_u=128,
        n_labels=4,
        dropout_rate=0.1,
        epochs=60,
        batch_size=32, 
        max_seq_len=512, # necessary to limit memory usage
        use_extracted_emb=use_extracted_emb, # set False if using BERT
    )

    # BERT wordpiece tokenizer
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=config.max_seq_len,
        do_lowercase=False,
    )
    def tokenizer(s: str, max_seq_len: int=config.max_seq_len) -> List[str]:
        return [Token(x) for x in token_indexer.wordpiece_tokenizer(s)[:max_seq_len]]

    # reading dataset
    reader = PairDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_percentage=config.use_percentage,
        use_extracted_emb=config.use_extracted_emb # set False if finetuning
    )

    train_ds = reader.read(working_dir)
    print(len(train_ds))

    # prepare batch
    vocab = Vocabulary.from_instances(train_ds) # the tutorial does not work (https://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/)
    iterator = BasicIterator(batch_size=config.batch_size)
    iterator.index_with(vocab) # this is a must for consistency reason

    # get model
    model = get_model(architecture=model_architecture, vocab=vocab, hyperparam=config, torch_device=torch_device)
    model.to(torch_device)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=0.001 if config.use_extracted_emb else 2e-5) # lr = 2e-5 for finetuning BERT, 0.001 if not
    train_start_time = time.time()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        shuffle=True, # to shuffle the batch
        cuda_device=cuda_device,
        should_log_parameter_statistics=False,
        num_epochs=config.epochs,
    )
    metrics = trainer.train()
    train_end_time = time.time()
    print('Finished Training, time %.3fmins' % ((train_end_time-train_start_time)/60.0))

    # # save model
    save_model(model, "test/test_run/run-1/")

    # # load model
    model2 = load_model("test/test_run/run-1/", torch_device)

    # test
    test_dir = "../data/ICNALE-SBERT/original/pairwise_link_labelling/test/"
    test_ds = reader.read(test_dir)
    print(len(test_ds))

    predictor = Predictor(model, iterator, cuda_device=cuda_device)
    test_preds, gold_preds = predictor.predict(test_ds)
    
    print(classification_report(y_true=gold_preds, y_pred=test_preds))
    print(confusion_matrix(y_true=gold_preds, y_pred=test_preds))
