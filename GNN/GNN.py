import os
import time
import argparse
from sklearn import metrics
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from utils import load_corpus
from metrics import *


class GCN(tf.keras.Model):
    def __init__(self, adj_matrix, layers, hidden_size, dropout=1.0):
        super(GCN, self).__init__()
        assert layers == len(
            hidden_size
        ), "'layers' input and 'hidden_size' do not match"

        self.hidden_size = hidden_size
        self.adj_matrix = adj_matrix
        self.dropout = dropout

        self.conv_layers = []
        for i in range(layers):
            self.conv_layers.append(Dense(units=self.hidden_size[i], activation="relu"))

    def call(self, X):

        if self.dropout == 1:
            conv = tf.linalg.matmul(self.adj_matrix, self.conv_layers[0](X))
            for layer in self.conv_layers[1:]:
                conv = tf.linalg.matmul(self.adj_matrix, layer(conv))
        else:
            drop = Dropout(self.dropout)
            conv = drop(self.conv_layers[0](X))
            conv = tf.linalg.matmul(self.adj_matrix, conv)
            for layer in self.conv_layers[1:]:
                conv = drop(layer(X))
                conv = tf.linalg.matmul(self.adj_matrix, conv)

        return conv


class TextGCN:
    def __init__(self, model, loss, optimizer, args):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.args = args

    def grad(self, X, y, train_mask, trainable_variables):
        with tf.GradientTape() as tape:
            loss_value = self.loss(self.model(X), y, train_mask)
        return loss_value.numpy(), tape.gradient(loss_value, trainable_variables)

    @staticmethod
    def one_run(model, X, y):
        return model(X)

    def train(self, X, y, train_mask):

        self.train_losses = []
        self.train_accuracy = []

        self.train_mask = train_mask

        print("Begin Learning...")

        for epoch in range(self.args.epochs):
            ctime = time.time()

            output = self.model(X)

            losses, grads = self.grad(output, y, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.train_losses.append(losses)
            accuracy = self.evaluate(output, y_train, train_mask)
            # accuracy = (tf.argmax(y_test,axis=-1) == tf.argmax(tf.nn.softmax(output,axis=0),axis=-1)).numpy().mean()
            self.train_accuracy.append(accuracy)

            print(f"Epoch {epoch+1} \t loss value : {losses} \t accuracy: {accuracy}")
            print(
                f"Time taken for epoch {epoch+1} : {round((time.time()-ctime)/60,2)} MINUTES"
            )

        # return self.model, train_losses, train_accuracy

    def evaluate(self, X, y, test_mask):
        return masked_accuracy(self.model(X), y, test_mask)
        # self.test_mask = test_mask
        # self.eval_output = self.model(X)
        # self.eval_softmax = tf.nn.softmax(self.eval_output,axis=0)
        # return tf.argmax(self.eval_softmax, axis=-1)


def run(args):
    (
        adj,
        features,
        y_train,
        y_val,
        y_test,
        train_mask,
        val_mask,
        test_mask,
        train_size,
        test_size,
    ) = load_corpus(args.select_data)

    train_mask = train_mask + val_mask
    y_train = y_train + y_val

    adj_dense = adj.todense().astype(np.float32)
    features_dense = features.todense().astype(np.float32)

    # features_train = features_dense[:-2189]
    # features_test = features_dense[-2189:]
    # y_train = y_train[:-2189]
    # y_test = y_test[-2189:]

    gcn_model = GCN(
        adj_dense,
        layers=args.layers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )

    loss_fn = masked_softmax_cross_entropy

    acc_fn = masked_accuracy

    optimizer = Adam(learning_rate=args.lr)

    model_textGCN = TextGCN(
        model=gcn_model, loss=loss_fn, optimizer=optimizer, args=args
    )

    model_textGCN.train(features_dense, y_train, train_mask)

    eval_result = model_textGCN.evaluate(features_dense, y_test, test_mask)

    print(f"Final Evaluation Result: {eval_result}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--select_data", default="R8", type=str, help="Select data to train"
    )

    parser.add_argument(
        "--layers", default=2, type=int, help="Define the number of layers"
    )

    parser.add_argument(
        "--hidden_size",
        default=[64, 8],
        type=int,
        nargs="+",
        help="Define output of each layer",
    )

    parser.add_argument(
        "--dropout", default=0.7, type=float, help="Define dropout rate"
    )

    parser.add_argument("--lr", default=1e-3, type=float, help="Set learning rate")

    parser.add_argument("")

    args = parser.parse_args()

    run(args)
