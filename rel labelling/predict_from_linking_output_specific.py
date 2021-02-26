"""
This script is to convert linking open automatic linking output prediction, then predict their relation labels, then convert the result to tsv
"""
import argparse
import ast
from os import listdir
from os.path import isfile, join
from typing import *
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util
from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.iterators import BasicIterator

from discourseunit import DiscourseUnit, Essay, NO_REL_SYMBOL
from treebuilder import TreeBuilder
from datasetreader import PairDatasetReader
from model_functions import *
from common_functions import *


class PairDatasetReaderSpecial(PairDatasetReader):
    def read_data(self, essay_code, source_target_sentences: list) -> Iterator[Instance]:
        """
        Args:
            source_target_sentences: list containing source and target sentences (per essay)

        Returns:
            Iterator[Instance]
        """
        for i in range(len(source_target_sentences)):
            yield self.text_to_instance(
                str(source_target_sentences[i][0]),
                str(source_target_sentences[i][1]),
                np.zeros(0),
                np.zeros(0),
                "sup", # the gold relation label does not mean anything here
                essay_code
            )

def get_dataset_reader():
    """
    Get dataset reader instance
    """
    # Fixed configuration
    config = Config(
        use_percentage=1.0, # how much data to use
        max_seq_len=512, # necessary to limit memory usage
        use_extracted_emb=False, # set False since we are using BERT
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
    reader = PairDatasetReaderSpecial(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_percentage=config.use_percentage,
        use_extracted_emb=config.use_extracted_emb # set False if finetuning
    )
    return config, reader


def open_essays(source_files):
    """
    Open essays into list of internal data structure

    Args:
        source_files (List[str]): list of filename to open

    Returns:
        list of Essay
    """
    essays_ann = []
    for source_file in source_files:
        essay = Essay(source_file)
        essays_ann.append(essay)
    return essays_ann


def load_best_BERT_run(best_run_dir):
    """
    Loading BERT model for prediction
    """
    # loading BERT model
    cuda_device = 0 if torch.cuda.is_available() else -1
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA device = "+str(cuda_device)+" (running on "+str(torch_device)+")")
    available_gpus = np.arange(torch.cuda.device_count()).tolist()
    print("Available GPUs "+str(available_gpus))

    print("loading BERT model (best run)")
    model = load_model(best_run_dir, torch_device)
    print("Model architecture:", model.param["class"])
    print("Model params", model.param)
    model.to(torch_device)

    return cuda_device, model


def create_pairwise_data(sentences, dist):
    """
    Create pairwise link labelling data

    Args:
        sentences (list[str])
        dist (list[int])

    Returns:
        list[tuple(str,str)]
    """
    rep = TreeBuilder(dist)
    component_labels = rep.auto_component_labels(AC_breakdown=True) 

    output = []
    for i in range(len(sentences)):
        if component_labels[i] == "non-arg comp.":
            pass
        else:
            if i+dist[i] != i: # the current sentence does not point to itself, i.e., not a root
                source = sentences[i]
                target = sentences[i+dist[i]]
                output.append((source, target))
    return output


def save_tsv(filename, essay_code, sentences, dist, label_preds):
    header = ["essay code", "unit id", "text", "target", "relation", "drop_flag"]
    rep = TreeBuilder(dist)
    component_labels = rep.auto_component_labels(AC_breakdown=True)

    f = open(filename, "w")
    f.write("\t".join(header)+"\n")
    label_idx = 0

    for i in range(len(sentences)):
        output_line = []
        output_line.append(essay_code)
        output_line.append(str(i+1))
        output_line.append(sentences[i])
        if component_labels[i] == "non-arg comp.":
            output_line.append("")
            output_line.append("")
            output_line.append("TRUE")
        else:
            target = i+1+dist[i]
            if target==i+1: # point to itself, i.e., root
                output_line.append("")
                output_line.append("")
            else: # not root
                output_line.append(str(target))
                output_line.append(label_preds[label_idx])
                label_idx += 1
            output_line.append("FALSE")

        f.write("\t".join(output_line)+"\n")
    assert(label_idx == len(label_preds))
    f.close()


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Convert linking output to tsv file')
    parser.add_argument(
        '-train_test_split', '--train_test_split', type=str, help='train_test_split_file)', required=False)
    parser.add_argument(
        '-test_dir', '--test_dir', type=str, help='dataset directory (test data in tsv)', required=True)
    parser.add_argument(
        '-pred_file', '--pred_file', type=str, help='directory containing the prediction result (specific run). Use run-13/link_preds.txt for the best Biaffine-SBERT model', required=True)
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, help='output directory', required=True)
    args = parser.parse_args()

    # get the list of test essays
    if args.train_test_split:
        supposed_code = []
        with open(args.train_test_split) as f:
            for line in f: 
                content = line.strip().split(",")
                essay_code = content[0].strip().split(".")[0][1:]
                verdict = content[1].strip()
                if verdict == "\"test\"":
                    supposed_code.append(essay_code)
        supposed_code.sort()
        print("# supposed test data", len(supposed_code))

    # open test essays
    files = list_files_in_dir(args.test_dir)
    files.sort()
    essays = open_essays(files)
    test_essays = []
    for i in range(len(essays)):
        # print(essays[i].essay_code+"\"")
        if args.train_test_split and (essays[i].essay_code in supposed_code):
            test_essays.append(essays[i])
        elif not args.train_test_split:
            test_essays.append(essays[i])
    print("# test essays ", len(test_essays))

    # open specific prediction file
    print("loading linking prediction file")
    with open(args.pred_file) as f:
        content = f.readlines()[0]
        prediction_dist = ast.literal_eval(content)

    # load model
    cuda_device, model = load_best_BERT_run("original/saved_models/BERT/run-11/")

    for i in range(len(test_essays)):
        
        sentences = test_essays[i].get_texts(order="original", normalised="True")
        dist = prediction_dist[i]
        
        # create pairwise data, then predict for this essay using BERT best run
        pairwise_data = create_pairwise_data(sentences, dist)
        config, reader = get_dataset_reader()
        test_ds = reader.read_data(test_essays[i].essay_code, pairwise_data)
        
        # iterator
        iterator = BasicIterator(batch_size=config.batch_size)
        iterator.index_with(model.vocab) # this is a must for consistency reason

        # predict
        predictor = Predictor(model, iterator, cuda_device=cuda_device)
        label_preds, _ = predictor.predict(test_ds) # gold preds here do not mean anything

        save_tsv(args.out_dir+test_essays[i].essay_code+".tsv", test_essays[i].essay_code, sentences, dist, label_preds)
        # input()




