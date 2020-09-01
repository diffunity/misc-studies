import os
import time
import argparse
import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from utils import load_corpus, preprocess_adj, preprocess_features
from metrics import *

class GCN(tf.keras.Model):
    def __init__(self, adj_matrix, layers, hidden_size, dropout=0.5):
        super(GCN, self).__init__()
        assert layers == len(hidden_size), "'layers' input and 'hidden_size' do not match"

        self.hidden_size = hidden_size
        self.adj_matrix = adj_matrix
        self.dropout = dropout

        self.conv_layer_1 = Dense(units=self.hidden_size[0], 
                                  activation="relu",
                                #   kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                #   activity_regularizer=tf.keras.regularizers.L2(0.001)
                                  )
        
        self.conv_layer_2 = Dense(units=self.hidden_size[1], 
                                  activation="relu",
                                #   kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                #   activity_regularizer=tf.keras.regularizers.L2(0.001)
                                  )
    
    # @tf.function
    def call(self, X):

        out = self.conv_layer_1(X)
        out = Dropout(self.dropout)(out)
        out = tf.matmul(self.adj_matrix, out)
        out = self.conv_layer_2(out)
        out = Dropout(self.dropout)(out)
        return tf.matmul(self.adj_matrix, out)

class TextGCN:
    def __init__(self, model, loss, optimizer, args):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.args = args

    @tf.function
    def grad(self, X, y, train_mask):
        
        with tf.GradientTape() as tape:
            # tape.watch(self.model.trainable_variables)
            output = self.model(X)
            loss_value = self.loss(output, y, train_mask)
        print("LOSS: ", loss_value)
        grad = tape.gradient(loss_value, self.model.trainable_variables)
        print("GRAD: ", grad)
        return loss_value, grad

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

            losses, grads = self.grad(X, y, train_mask)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.train_losses.append(losses)
            accuracy = self.evaluate(X, y, train_mask)
            self.train_accuracy.append(accuracy)

            print(
                f"Epoch {epoch+1} \t loss value : {losses} \t accuracy: {accuracy}"
            )
            print(
                f"Time taken for epoch {epoch+1} : {round((time.time()-ctime)/60,2)} MINUTES"
            )
        
        # return self.model, self.train_losses, train_accuracy
    
    def predict(self, X):
        return tf.nn.softmax(self.model(X))

    def evaluate(self, X, y, test_mask):
        return masked_accuracy(self.model(X), y, test_mask)

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


    adj_dense = preprocess_adj(adj).toarray().astype(np.float32)
    features_dense = preprocess_features(features).toarray().astype(np.float32)

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    train_mask = train_mask.astype(np.float32)
    test_mask = test_mask.astype(np.float32)

    gcn_model = GCN(
        tf.convert_to_tensor(adj_dense),
        layers=args.layers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )

    loss_fn = masked_softmax_cross_entropy

    # acc_fn = masked_accuracy

    optimizer = Adam(learning_rate=args.lr)
    # print("Model Layers: ", gcn_model.trainable_variables)
    model_textGCN = TextGCN(
        model=gcn_model, loss=loss_fn, optimizer=optimizer, args=args
    )

    model_textGCN.train(features_dense, y_train, train_mask)

    sns.distplot(model_textGCN.train_accuracy)
    plt.savefig("train_acc.png")
    
    plt.clf()

    sns.distplot(model_textGCN.train_losses)
    plt.savefig("train_losses.png")

    eval_result = model_textGCN.evaluate(features_dense, y_test, test_mask)

    print(f"Final Evaluation Result: {eval_result}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--select_data", 
                        default="R8", 
                        type=str, 
                        help="Select data to train"
                        )

    parser.add_argument("--layers", 
                        default=2, 
                        type=int, 
                        help="Define the number of layers"
                        )

    parser.add_argument("--hidden_size",
                        default=[64, 8],
                        type=int,
                        nargs="+",
                        help="Define output of each layer",
                        )

    parser.add_argument("--dropout", 
                        default=0.5, 
                        type=float, 
                        help="Define dropout rate"
                        )

    parser.add_argument("--lr", 
                        default=0.02, 
                        type=float, 
                        help="Set learning rate"
                        )

    parser.add_argument("--epochs", 
                        default=500, 
                        type=int, 
                        help="Training epochs"
                        )

    args = parser.parse_args()

    run(args)
