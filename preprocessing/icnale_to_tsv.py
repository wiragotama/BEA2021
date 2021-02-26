"""
by  Jan Wira Gotama Putra

This script is used to convert annotated essays (in HTML) to tsv
"""
import os
import sys
import numpy as np
import csv
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from discourseunit import DiscourseUnit
from discourseunit import Essay
from discourseunit import NO_REL_SYMBOL
import matplotlib.pyplot as plt
import argparse
from common_functions import list_files_in_dir
from common_functions import open_essays

if __name__ == "__main__":
    # user arguments
    parser = argparse.ArgumentParser(description='HTML (annotated essays) to TSV converter')
    parser.add_argument(
        '-in_dir', '--in_dir', type=str, help='relative directory of corpus (html files)', required=True)
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, help='relative directory for output (tsv files)', required=True)
    parser.add_argument(
        '-essay_list', '--essay_list', type=str, help='csv file containing the supposed essay list', required=True)
    args = parser.parse_args()
    directory = args.in_dir

    # open files
    files = list_files_in_dir(directory)
    files.sort()
    essays = open_essays(files)

    # open supposed essay list
    supposed_list = []
    with open(args.essay_list) as f:
        for line in f: 
            essay_code = line.strip()
            supposed_list.append(essay_code)
    supposed_list.sort()
    print("# supposed essays", len(supposed_list))

    # convert to tsv and then save
    saved = []
    for essay in essays:
        save_path = args.out_dir + essay.essay_code + ".tsv"

        # consistency checking
        if save_path in saved:
            print (">>>> two essays found!!!", essay.essay_code)
        else:
            saved.append(save_path)
            f = open(save_path, 'w+') # overwrite
            f.write(essay.to_tsv())
            f.close()

            try:
                supposed_list.remove(essay.essay_code.strip())
            except:
                print(">>>> not in the list!!", essay.essay_code.strip())
            

    print("# Remaining essays to be parsed")
    for x in supposed_list:
        print(x)