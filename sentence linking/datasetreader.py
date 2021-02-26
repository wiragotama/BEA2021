"""
by  Jan Wira Gotama Putra

Given a sequence of sentences, find the distance from source to target sentence
"""
import ast
import numpy as np
from os import listdir
from os.path import isfile, join
from abc import ABC, abstractmethod
from copy import deepcopy
import random
import torch
from torch import tensor
import scipy
from typing import *
from functools import partial
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util

from allennlp.data.fields import TextField, MetadataField, ArrayField, LabelField, SequenceLabelField, ListField, MultiLabelField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.iterators import BasicIterator, BucketIterator


def tonp(tsr): return tsr.detach().cpu().numpy() # for seamless interaction between gpu and cpu


class SeqDatasetReader(DatasetReader):
    """
    Dataset structure for linking experimental setting
    An instance represents an essay
    """

    def __init__(self, tokenizer: Callable[[str], List[str]],
                token_indexers: Dict[str, TokenIndexer],
                use_extracted_emb: bool,
                mode: str,
                use_percentage: float=1.0) -> None: 
        """
        Args:
            tokenizer: tokenizer object
            token_indexer: token indexer object
            use_extracted_emb (bool): if we want to store previously extracted sentence embeddings (preprocessing). Set this False in case of FineTuning
            mode (str): "STL" (single task learning) or "MTL" (multi-task learning)
            use_percentage (float): how many (percent) of the essays you want to use (this is useful for cutting the number of samples for test run); must be between 0 and 1
        """
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if mode == "dep":
            mode = "STL"
            print("Load data in STL mode for dependency-parsing setting")
        if use_percentage > 1 or use_percentage < 0:
            raise Exception("[ERROR] use_percentage variable must be between [0,1]")
        if not mode in {"STL", "MTL"}:
            raise Exception("[ERROR] mode variable must be 'STL' or 'MTL'")
        self.use_percentage = use_percentage
        self.use_extracted_emb = use_extracted_emb
        self.ref_rel_dists = [] # output distances (labels) in the dataset
        self.mode = mode
        if self.mode == "MTL":
            self.ref_component_labels = [] # component labels in the dataset


    @overrides
    def text_to_instance(self,
                        sentences: np.ndarray,
                        sent_embeddings: np.ndarray, 
                        component_labels: np.ndarray,
                        rel_dists: np.ndarray,
                        essay_code: str,
                        seq_len: int) -> Instance:
        """
        Args:
            sentences (np.ndarray): source sentence (not tokenized)
            sent_embeddings (np.ndarray): target sentence (not tokenized)
            component_labels (np.ndarray): source sentence embedding (pre-processed)
            rel_dists (np.ndarray): target sentence embedding (pre-processed)
            essay_code (str): essay code of the corresponding Instance
            seq_len (int): sequence length of the current instance

        Returns:
            Instance
        """
        # meta-data
        essay_code_field = MetadataField(essay_code)
        fields = {"essay_code": essay_code_field}

        # to be used for sequence mask
        # during training, we will have (batch_size, seq_len (# sentences), token_length)
        # since the number of sentences are different for each essay, we need some masking when feeding it into the network
        seq_len_field = MetadataField(seq_len)
        fields["seq_len"] = seq_len_field
        
        # use extracted embedding
        if self.use_extracted_emb:
            list_emb_field = []
            for emb in sent_embeddings:
                list_emb_field.append(ArrayField(emb))
            list_emb_field = ListField(list_emb_field)
            fields["sent_embeddings"] = list_emb_field
            
            ref_seq = list_emb_field # required for sequence label field

        # use raw text
        else:
            list_text_field = []
            for sentence in sentences:
                if self.tokenizer != None:
                    list_text_field.append(TextField(self.tokenizer(sentence), self.token_indexers))
                else:
                    list_text_field.append(MetadataField(sentence))
            sentence_field = ListField(list_text_field)
            fields["sentences"] = sentence_field

            ref_seq = sentence_field # required for sequence label field

        # dist
        rel_dist_field = SequenceLabelField(rel_dists, label_namespace="rel_dist_labels", sequence_field=ref_seq) # automatic padding uses "0" as padding
        fields["rel_dists"] = rel_dist_field

        # label
        if self.mode == "MTL":
            rel_label_field = SequenceLabelField(component_labels, label_namespace="component_labels", sequence_field=ref_seq) # automatic padding uses "0" as padding
            fields["component_labels"] = rel_label_field

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
                sentences, sent_embeddings, component_labels, rel_dists = self.__open_essay(directory, essay_code)
                yield self.text_to_instance(
                    sentences,
                    sent_embeddings,
                    component_labels,
                    rel_dists,
                    essay_code,
                    len(sentences)
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
            if not ((x + ".rel_distances" in filecodes) and
                    (x + ".component_labels" in filecodes) and
                    (x + ".sentences" in filecodes) and
                    (x + ".vectors" in filecodes)):
                flag = False
                raise Exception("[Error] essay", x, "has incomplete files")

        # for ease of debugging
        unique_names = list(unique_names)
        unique_names.sort()

        return flag, unique_names


    def __open_essay(self, directory: str, essay_code: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Open essay information

        Args:
            directory (str)
            essay_code (str)

        Returns:
            {
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray,
                numpy.ndarray
            }
        """
        text_file = directory + essay_code + ".sentences"
        emb_file = directory + essay_code + ".vectors"
        component_file = directory + essay_code + ".component_labels"
        rel_dist = directory + essay_code + ".rel_distances"
        
        with open(text_file, 'r') as f:
            sentences = np.array(ast.literal_eval(f.readline()), dtype=str)
        with open(emb_file, 'r') as f:
            sent_embeddings = np.array(ast.literal_eval(f.readline()), dtype=float)
        with open(component_file, 'r') as f:
            component_labels = np.array(ast.literal_eval(f.readline()), dtype=str)
        with open(rel_dist, 'r') as f:
            rel_dists = np.array(ast.literal_eval(f.readline()), dtype=str)
        
        # checking
        assert(len(sentences) == len(component_labels))
        assert(len(sent_embeddings) == len(component_labels))
        assert(len(sent_embeddings) == len(rel_dists))
       
        return sentences, sent_embeddings, component_labels, rel_dists


    @staticmethod
    def get_text_field_mask(token_batch: torch.Tensor) -> torch.Tensor:
        """
        The attention mask length does not match with wordpiece tokenizer, so this is the solution

        Args:
            token_batch (Tensor): of (batch_size, seq_len, tokens); text already split and then converted to their indices
        """
        token_batch = tonp(token_batch)
        batch_mask = []
        for essay in token_batch:
            essay_mask = []
            for sentence in essay:
                sentence_mask = []
                for token in sentence:
                    if token.item() == 0: # padding
                        sentence_mask.append(0)
                    else:
                        sentence_mask.append(1)
                essay_mask.append(sentence_mask)
            batch_mask.append(essay_mask)

        return torch.LongTensor(np.array(batch_mask))


    @staticmethod
    def get_essay_mask(essay: torch.Tensor) -> torch.Tensor:
        """
        The attention mask length does not match with wordpiece tokenizer, so this is the solution

        Args:
            essay (Tensor): of (seq_len, tokens); text already split and then converted to their indices
        """
        token_batch = tonp(essay)
        essay_mask = []
        for sentence in essay:
            sentence_mask = []
            for token in sentence:
                if token.item() == 0: # padding
                    sentence_mask.append(0)
                else:
                    sentence_mask.append(1)
            essay_mask.append(sentence_mask)

        return torch.LongTensor(np.array(essay_mask))


    @staticmethod
    def get_batch_seq_mask(seq_len: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for feeding batch of sequences into LSTM

        Args:
            seq_len (Tensor): of (batch_size); containing the correct sequence length (without padding)
        """
        max_len = max(seq_len)
        batch_mask = []
        for x in seq_len:
            one = [1] * x
            pads = [0] * (max_len-x)
            seq_mask = one + pads
            batch_mask.append(seq_mask)

        return torch.FloatTensor(batch_mask)



"""
Example
"""
if __name__ == "__main__": 
    working_dir = "../data/ICNALE/original/linking/train/"
    use_extracted_emb = True # set False if using raw text (finetuning case)
    setting = "MTL"
    
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
    reader = SeqDatasetReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer},
        use_extracted_emb=use_extracted_emb, # set False if using raw text (finetuning case)
        mode=setting,
        use_percentage=1.0,
    )

    train_ds = reader.read(working_dir)
    print(len(train_ds))
    if use_extracted_emb:
        print(vars(train_ds[0].fields["sent_embeddings"][0]))
    else:
        print(vars(train_ds[0].fields["sentences"][0]))
    if setting == "MTL":
        print(vars(train_ds[0].fields["component_labels"]))
    print(vars(train_ds[0].fields["rel_dists"]))
    print(vars(train_ds[0].fields["essay_code"]))
    print(vars(train_ds[0].fields["seq_len"]))

    # prepare vocabulary
    vocab = Vocabulary.from_instances(train_ds) # somehow, this is a must

    # batch-ing
    iterator = BasicIterator(batch_size=8)
    iterator.index_with(vocab) # this is a must for consistency reason
    batch = next(iter(iterator(train_ds, shuffle=False))) # shuffle = False just for checking
    # print(batch)
    if use_extracted_emb:
        print("embeddings", batch['sent_embeddings'].shape)
    else:
        print("sentence", batch['sentences']['tokens'].shape)
        print("mask", batch['sentences']['mask'].shape) # this is problematic because in allennlp, the shape is -2 (in length) of tokens
    if setting == "MTL":
        print("component_labels", batch['component_labels'].shape)
    print("rel_dists", batch['rel_dists'].shape)
    print("seq_len", batch['seq_len'])

    # token masking problem
    if not use_extracted_emb:
        # to solve token masking problem
        correct_mask = SeqDatasetReader.get_text_field_mask(batch['sentences']['tokens'])
        # print(correct_mask)
        print("correct mask", correct_mask.shape)

    # sequence mask
    sequence_mask = SeqDatasetReader.get_batch_seq_mask(batch['seq_len'])
    # print(sequence_mask)
    print("sequence mask", sequence_mask.shape)

    # mapping label back
    mapping_trial = batch['rel_dists'][0]
    for i in range(len(mapping_trial)):
        if sequence_mask[0][i]==1:
            print(vocab.get_token_from_index(int(mapping_trial[i]), namespace="rel_dist_labels"))

    # test save
    vocab.save_to_files("vocabulary/")

