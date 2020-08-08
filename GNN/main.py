import os
import argparse
import tensorflow as tf
from preprocess import data_generator
from GNN import GCN, TextGCN

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs",
                        default=5,
                        type=int,
                        help="Number of training epochs")

    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="Training batch size")

    parser.add_argument("--gradient_accumulation_steps",
                        default=8,
                        type=int,
                        help="Gradient accumulation steps")


    
    args = parser.parse_args()
