"""
Author: Jan Wira Gotama Putra

Using SBERT model to encoder a sentence

References: (don't forget to cite!)
- https://github.com/UKPLab/sentence-transformers
"""
from typing import *
import numpy as np
import torch
from sentence_transformers import SentenceTransformer



class SBERTencoder:
    """
    Sentence encoder using SBERT
    """

    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')


    def text_to_vec(self, l: List[str]) -> List[List[float]]:
        """
        Convert a list of sentences into their vector form. 
        SBERT automatically tokenize and converts the text into lowercase (for UNCASED)
        
        Args:
            l (List[str]): list of str (sentence) to be converted
        
        Returns:
            List[List[float]], denoting sentences encoded in their vector form
        """
        vector = np.array(self.model.encode(l))
        return vector


if __name__ == "__main__":
    encoder = SBERTencoder() # set the BERTserver first
    vector = encoder.text_to_vec(["Say goodbye", "I want to eat cookies"])
    print(vector.shape)
    print(type(vector))
    print(vector)
