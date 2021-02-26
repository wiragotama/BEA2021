"""
Author: Jan Wira Gotama Putra

Using BERT model to encoder a sentence

References: (don't forget to cite!)
- https://github.com/google-research/bert#pre-trained-models
- https://github.com/hanxiao/bert-as-service
"""
from typing import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient
import copy


class BERTencoder:
    """
    Sentence encoder using BERT (client side)
    """

    def __init__(self, port=5555, port_out=5556):
        """
        Word vocabulary initialization

        Args:
            port (int, optional, defaults to 5555): server port for receiving data from client
            port_out(int, optional, defaults to 5556): server port for sending result to client
        """
        self.__client = BertClient(port=port, port_out=port_out)


    def text_to_vec(self, l: List[str]) -> np.ndarray:
        """
        Convert a list of sentences into their vector form. 
        Counted by the average (default) of word vectors (composing the sentence)
        BERT automatically tokenize and converts the text into lowercase (for UNCASED)
        
        Args:
            l (List[str]): list of str (sentence) to be converted
        
        Returns:
            np.ndarray, denoting sentences encoded in their vector form
        """
        vector = self.__client.encode(l)
        return vector


if __name__ == "__main__":
    encoder = BERTencoder() # set the BERTserver first
    vector = encoder.text_to_vec(["Say goodbye", "I want to eat cookies"])
    print(vector.shape)
    print(type(vector))
    print(vector)
