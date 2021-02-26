"""
Author: Jan Wira Gotama Putra
"""
from os import listdir
from os.path import isfile, join
from discourseunit import Essay
import numpy as np
from typing import *


def list_files_in_dir(directory):
    """
    List all files in a directory and remove `.DS_Store`

    Args:
        directory (str): path to directory

    Returns:
        list of filename with `.DS_Store` removed
    """
    annotation = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    annotation = remove_unwanted_files(annotation)
    annotation.sort()
    return annotation


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


def remove_unwanted_files(files, extensions=['.DS_Store', '.gitignore']):
    """
    Remove file (with unwanted extensions) from list of filenames

    Args:
        files (List[str]): denoting list of filename
        extensions (List[str]): denoting unwanted extensions
    
    Returns:
        list of filename with `.DS_Store` removed
    """
    for source_file in files:
        for ex in extensions:
            if ex in source_file:
                files.remove(source_file)
                break
    return files


def print_stats(stats_name, info, print_total=True):
    """
    print statistics of the supplied vals

    Args:
        stats_name (str): name of the statistics
        vals (List[float]): values
        print_total (bool): whether to print the sum of the values
    """
    print("    %s \t\t%.0f \t%.0f \t%.0f \t%.1f \t%.1f" % 
    (
        stats_name,
        np.sum(info) if print_total else np.nan,
        np.max(info),
        np.min(info),
        np.average(info),
        np.std(info),
    ))