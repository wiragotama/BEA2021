"""
by  Jan Wira Gotama Putra

This is the script to load the dataset for pairwise link labelling

Experimental setting:
Given a pair of source and target sentence <s, t> (already linked), find the label of the relation from source to target sentence
"""
import ast
import numpy as np
from os import listdir
from os.path import isfile, join
from abc import ABC, abstractmethod
from copy import deepcopy
import random
import torch
import scipy
from typing import *
from functools import partial
from overrides import overrides

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


class PairDatasetReader(DatasetReader):
    """
    Dataset structure for pairwise link labelling experimental setting
    """

    REF_LABELS = ["=", "att", "det", "sup"] # DO NOT CHANGE THE ORDER (sorted lexicographically)!

    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                token_indexers: Dict[str, TokenIndexer] = None,
                use_percentage: float=1.0,
                use_extracted_emb: bool=True) -> None:
        """
        Args:
            tokenizer: tokenizer object, assign None if you do not want to tokenize sentences
            token_indexer: token indexer object
            use_percentage (float): how many (percent) of the essays you want to use (this is useful for cutting the number of samples for test run); must be between 0 and 1
            use_extracted_emb (bool): if we want to store previously extracted sentence embeddings (preprocessing). Set this False in case of FineTuning
        """
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if use_percentage > 1 or use_percentage < 0:
            raise Exception("[ERROR] use variable must be between [0,1]")
        self.use_percentage = use_percentage
        self.use_extracted_emb = use_extracted_emb


    @overrides
    def text_to_instance(self,
                        source_sent: str,
                        target_sent: str, 
                        source_emb: np.ndarray,
                        target_emb: np.ndarray,
                        label: str,
                        essay_code: str=None) -> Instance:
        """
        Args:
            source_sent (str): source sentence (not tokenized)
            target_sent (str): target sentence (not tokenized)
            source_emb (np.ndarray): source sentence embedding (pre-processed)
            target_emb (np.ndarray): target sentence embedding (pre-processed)
            label (str): relation label
            essay_code (str): essay code of the corresponding Instance

        Returns:
            Instance
        """
        # essay code
        essay_code_field = MetadataField(essay_code)
        fields = {"essay_code": essay_code_field}

        # source and target sentences
        if not self.use_extracted_emb:
            if self.tokenizer != None:
                source_sent = self.tokenizer(source_sent)
                target_sent = self.tokenizer(target_sent)
                sentence_pair = source_sent + [Token("[SEP]")] + target_sent
                source_target_sent = TextField(sentence_pair, self.token_indexers)
                fields["source_target_sent"] = source_target_sent # sentence pair
            else:
                source_sent = MetadataField(source_sent)
                target_sent = MetadataField(target_sent)
                fields["source_sent"] = source_sent
                fields["target_sent"] = target_sent
        
        # source and target embeddings
        if self.use_extracted_emb:
            source_emb_field = ArrayField(array=source_emb)
            fields["source_emb"] = source_emb_field
            target_emb_field = ArrayField(array=target_emb)
            fields["target_emb"] = target_emb_field
        
        # label
        label_field = LabelField(label)
        fields["label"] = label_field

        return Instance(fields)
    

    @overrides
    def _read(self, directory: str) -> Iterator[Instance]:
        """
        Args:
            directory (str): containing the dataset
            use (float): how many (percent) of the essays you want to use (this is useful for cutting the number of samples for test run); must be between 0 and 1
        
        Returns:
            Iterator[Instance]
        """
        # file checking
        source_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        source_files.sort()
        flag, essay_codes = self.__source_files_checking(source_files)

        # open files if passed the checking
        if flag:
            if self.use_percentage < 1:
                essay_codes = random.sample(essay_codes, int(len(essay_codes) * self.use_percentage))
                essay_codes.sort()

            # read all files in the directory
            for essay_code in essay_codes:
                source_target_sentences, sent_embeddings, rel_labels = self.__open_essay(directory, essay_code)
                for i in range(len(source_target_sentences)):
                    yield self.text_to_instance(
                        str(source_target_sentences[i][0]),
                        str(source_target_sentences[i][1]),
                        sent_embeddings[i][0],
                        sent_embeddings[i][1],
                        str(rel_labels[i]),
                        essay_code
                    )


    def __source_files_checking(self, source_files: List[str]) -> (bool, List[str]):
        """
        Check whether the source files is complete
        Definition: each essay is represented by two files 
            - ".source_target_sentences"
            - ".source_target_sentences_embedding" (source and target texts already in embedding form)
            - ".source_target_rels" (relation between source -> target)

        Check if each unique essay (according to filename) has those three files

        Args:
            source_files (:obj:`list` of :obj:`str`)
        
        Returns:
            bool (True or False)
            List[str], unique filenames
        """
        # get all unique essay codes and existing files
        unique_names = set()
        filecodes = []
        for x in source_files:
            if (".DS_Store" not in x) and (".gitignore" not in x):
                filecode = x.split("/")[-1]
                essay_code = filecode.split(".")[0]

                unique_names.add(essay_code)
                filecodes.append(filecode)

        # check if for each essay code, there are three corresponding files 
        flag = True
        for x in unique_names:
            if not ((x + ".source_target_sentences" in filecodes) and
                    (x + ".source_target_rels" in filecodes) and
                    (x + ".source_target_sentences_embedding" in filecodes)):
                flag = False
                raise Exception("[Error] essay", x, "has incomplete files")

        # for ease of debugging
        unique_names = list(unique_names)
        unique_names.sort()

        return flag, unique_names


    def __open_essay(self, directory: str, essay_code:str) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Open essay information

        Args:
            directory (str)
            essay_code (str)

        Returns:
            {
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray
            }
        """
        text_file = directory + essay_code + ".source_target_sentences"
        emb_file = directory + essay_code + ".source_target_sentences_embedding"
        rel_file = directory + essay_code + ".source_target_rels"
        
        with open(text_file, 'r') as f:
            sentences = np.array(ast.literal_eval(f.readline()), dtype=str)
        with open(emb_file, 'r') as f:
            sent_embeddings = np.array(ast.literal_eval(f.readline()), dtype=float)
        with open(rel_file, 'r') as f:
            rel_labels = np.array(ast.literal_eval(f.readline()), dtype=str)
        
        # checking
        assert(len(sentences) == len(rel_labels))
        assert(len(sent_embeddings) == len(rel_labels))
       
        return sentences, sent_embeddings, rel_labels


"""
Example
"""
if __name__ == "__main__": 
    working_dir = "../data/ICNALE/original/pairwise_link_labelling/test/"
    
    # BERT wordpiece tokenizer
    max_seq_len = 512 # maximum number of tokens when tokenizing a text
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-multilingual-cased", # recommended by Google
        max_pieces=max_seq_len,
        do_lowercase=False,
    )

    def tokenizer(s: str, max_seq_len: int=512) -> List[str]:
        return [Token(x) for x in token_indexer.wordpiece_tokenizer(s)[:max_seq_len]]

    # reading dataset
    reader = PairDatasetReader(
        tokenizer=None, # None if you do not want to tokenize
        token_indexers={"tokens": token_indexer},
        use_percentage=0.1,
        use_extracted_emb=False # set False if finetuning
    )

    train_ds = reader.read(working_dir)
    print(len(train_ds))
    # print(vars(train_ds[0].fields["source_target_sent"]))
    print(vars(train_ds[0].fields["source_sent"]))
    print(vars(train_ds[0].fields["target_sent"]))
    # print(vars(train_ds[0].fields["source_emb"]))
    # print(vars(train_ds[0].fields["target_emb"]))
    print(vars(train_ds[0].fields["label"]))
    print(vars(train_ds[0].fields["essay_code"]))

    # prepare vocabulary
    vocab = Vocabulary.from_instances(train_ds) # the tutorial does not work (https://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/)

    # batch-ing
    iterator = BasicIterator(batch_size=2)
    iterator.index_with(vocab) # this is a must for consistency reason
    batch = next(iter(iterator(train_ds, shuffle=True)))
    print(batch)