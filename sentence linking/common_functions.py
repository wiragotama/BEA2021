"""
Author: Jan Wira Gotama Putra
"""
from os import listdir
from os.path import isfile, join
from typing import *

def list_files_in_dir(directory: str):
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


def remove_unwanted_files(files: List[str], extensions: List[str]=['.DS_Store', '.gitignore']):
    """
    Remove file (with unwanted extensions) from list of filenames

    Args:
        files (:obj:`list` of :obj:`str`): denoting list of filename
        extensions (:obj:`list` of :obj:`str`): denoting unwanted extensions
    
    Returns:
        list of filename with `.DS_Store` removed
    """
    for source_file in files:
        for ex in extensions:
            if ex in source_file:
                files.remove(source_file)
                break
    return files
