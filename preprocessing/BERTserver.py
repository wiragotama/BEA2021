"""
Author: Jan Wira Gotama Putra

Using BERT model to encoder a sentence

References: (don't forget to cite!)
- https://github.com/google-research/bert#pre-trained-models
- https://github.com/hanxiao/bert-as-service
"""
import time
import os
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
import argparse


class BERTserver:
    """
    Sentence encoder using BERT (server side)
    """

    def __init__(self, path,  port='5555', port_out='5556', pooling_strategy='REDUCE_MEAN'):
        """
        Word vocabulary initialization
        
        Args:
            path (str): BERT pretrained vector path
            port (str, optional, defaults to '5555'): server port for receiving data from client
            port_out(str, optional, defaults to '5556'): server port for sending result to client
            pooling_strategy(str, optional, defaults to `REDUCE_MEAN`): {NONE, REDUCE_MAX, REDUCE_MEAN, REDUCE_MEAN_MAX, FIRST_TOKEN, LAST_TOKEN}
        """
        self.__port = port
        self.__port_out = port_out
        args = get_args_parser().parse_args(['-model_dir', path,
                                     '-port', self.__port,
                                     '-port_out', self.__port_out,
                                     '-max_seq_len', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu',
                                     '-pooling_strategy', pooling_strategy])
        self.__server = BertServer(args)
        self.__server.start()


def print_args(port, port_out, pooling, model):
    print("[LOG BERTserver] port", port)
    print("[LOG BERTserver] port_out", port_out)
    print("[LOG BERTserver] pooling", pooling)
    print("[LOG BERTserver] model", model)


if __name__ == "__main__":
    # set the correspoinding pooling considering BERTencoder
    parser = argparse.ArgumentParser(description='BERT server')
    parser.add_argument(
        '-port', '--port', type=int, help='port (default 5555)', required=False)
    parser.add_argument(
        '-port_out', '--port_out', type=int, help='port_out (default 5556)', required=False)
    parser.add_argument(
        '-pooling', '--pooling', type=str, help='pooling strategy: {NONE, REDUCE_MAX, REDUCE_MEAN, REDUCE_MEAN_MAX, FIRST_TOKEN, LAST_TOKEN} (default REDUCE_MEAN)', required=False)
    parser.add_argument(
        '-bert_dir', '--bert_dir', type=str, help='directory of pre-trained bert model', required=True)

    args = parser.parse_args()
    print_args(args.port, args.port_out, args.pooling, args.bert_dir)
    print()

    if args.pooling!=None:
        bertserver = BERTserver(args.bert_dir, pooling_strategy=args.pooling)
    else:
        pooling_default = "REDUCE_MEAN"
        print("[LOG BERTserver] pooling unspecified, use", pooling_default,"\n")
        bertserver = BERTserver(args.bert_dir, pooling_strategy=pooling_default)
