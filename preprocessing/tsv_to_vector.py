"""
by  Jan Wira Gotama Putra

This script is used to convert tsv files into their vector form (e.g., vector of sentences etc.)
"""
import argparse
import numpy as np
import csv
from common_functions import list_files_in_dir
from common_functions import open_essays
from BERTencoder import BERTencoder
from SBERTencoder import SBERTencoder
from treebuilder import TreeBuilder


def check_train_or_test(info, query):
    """
    Check whether the query is train or test data

    Args:
        info (:obj:`list` of :obj:`list` of :obj:`str`): split information
        query (str) essay_code
    
    Returns:
        int: index of query in the info
    """
    for i in range(len(info)):
        if info[i][0] == query:
            return info[i][1]
    return None # not found


def save_content_to_file(filepath, content):
    """
    Args:
        filepath (str)
        content (list)
    """
    f = open(filepath, 'w+') # overwrite
    f.write(str(content))
    f.close()


if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='Convert TSV (annotated essays) to their vector forms')
    parser.add_argument(
        '-in_dir', '--in_dir', type=str, help='relative directory of corpus (tsv files)', required=True)
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, help='relative directory for output (vector form)', required=True)
    parser.add_argument(
        '-split', '--split', type=str, help='train_test_split information file', required=False)
    parser.add_argument(
        '-use_reordered', '--use_reordered', help='specify whether to use reordered and repaired text', action='store_true')
    parser.add_argument(
        '-encoder', '--encoder', type=str, help='BERT or SBERT', required=True)
    args = parser.parse_args()
    
    # open files
    files = list_files_in_dir(args.in_dir)
    essays = open_essays(files)
    print("Data:",len(essays), "essays")

    # encoder
    if args.encoder == "BERT":
        encoder = BERTencoder() # optioally: replace using transformer-based LM from huggingface
    elif args.encoder == "SBERT":
        encoder = SBERTencoder()
    else:
        raise Exception("encoder not known!")

    # train test split
    if args.split:
        print ("Split provided")
        with open(args.split, 'r') as f:
            split_info = [row for row in csv.reader(f.read().splitlines(), delimiter=',')]
        for i in range(len(split_info)):
            split_info[i][0] = split_info[i][0].split(".")[0] # delete the file extension
            split_info[i][1] = split_info[i][1].replace('\"','').strip()

    # use original (without text repair) or reordered text (with text repair)
    if args.use_reordered:
        inp_order = "reordering"
        inp_normalised = False # use revised text
    else:
        inp_order = "original"
        inp_normalised = True # use original text

    # convert to vector form
    for it in range(len(essays)):
        essay = essays[it]
        print("Processing",essay.essay_code)

        ################
        # linking task #
        ################
        sentences = essay.get_texts(order=inp_order, normalised=inp_normalised) # input feature
        rel_distances, rel_labels = essay.get_rel_distances(mode=inp_order, include_non_arg_units=True) # prediction label
        vectors = encoder.text_to_vec(sentences) # convert to sentence embedding
        print("  > ", it+1, essay.essay_code, vectors.shape)

        # assertion to check whether we have included non-arg-units here
        assert (len(sentences) == len(rel_distances))
        assert (len(sentences) == len(rel_labels))

        # determine where to save the file
        if args.split:
            split_folder = check_train_or_test(split_info, essay.essay_code)
            assert (split_folder != None)
            split_folder = split_folder + "/"
        else: 
            split_folder = "" # no split information provided

        # component labels
        rep = TreeBuilder(rel_distances)
        component_labels = rep.auto_component_labels(AC_breakdown=True)

        # save to file
        save_content_to_file(args.out_dir + "linking/" + split_folder.lower() + essay.essay_code + ".sentences", sentences)
        save_content_to_file(args.out_dir + "linking/" + split_folder.lower() + essay.essay_code + ".vectors", vectors.tolist())
        save_content_to_file(args.out_dir + "linking/" + split_folder.lower() + essay.essay_code + ".rel_distances", rel_distances)
        # save_content_to_file(args.out_dir + "linking/" + split_folder.lower() + essay.essay_code + ".rel_labels", rel_labels)
        save_content_to_file(args.out_dir + "linking/" + split_folder.lower() + essay.essay_code + ".component_labels", component_labels)


        # #######################
        # # link labelling task #
        # #######################
        source_target_sentences, source_target_rels = essay.get_pairwise_link_labelling_data(normalised=inp_normalised)
        source_target_sentences_embedding, source_target_rels = essay.get_pairwise_link_labelling_data(encode_sentences=True, encoder=encoder, normalised=inp_normalised)
        save_content_to_file(args.out_dir + "pairwise_link_labelling/" + split_folder.lower() + essay.essay_code + ".source_target_sentences", source_target_sentences)
        save_content_to_file(args.out_dir + "pairwise_link_labelling/" + split_folder.lower() + essay.essay_code + ".source_target_sentences_embedding", source_target_sentences_embedding) # for a baseline
        save_content_to_file(args.out_dir + "pairwise_link_labelling/" + split_folder.lower() + essay.essay_code + ".source_target_rels", source_target_rels)

