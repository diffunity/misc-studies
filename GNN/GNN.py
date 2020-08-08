import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import time

class GCN(tf.keras.Model):
    def __init__(self, adj_matrix, layers, hidden_size):
        super(GCN, self).__init__()
        assert layers==len(hidden_size), "'layers' input and 'hidden_size' do not match"
        
        self.hidden_size = hidden_size
        self.adj_matrix = adj_matrix
        
        self.conv_layers = []
        for i in range(layers):
            self.conv_layers.append(
                tf.keras.layers.Dense(units=self.hidden_size[i], activation="relu")
            )

    def call(self, X):
        
        conv = tf.linalg.matmul(self.adj_matrix, self.conv_layers[0](X))
        
        for layer in self.conv_layers[1:]:
            conv = tf.linalg.matmul(self.adj_matrix, layer(conv))

        return conv

class TextGCN():

    def __init__(self, model, loss, optimizer, data_generator, args):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.data_generator = data_generator
        self.args = args

    def grad(self, X, y, trainable_variables):
        with tf.GradientTape() as tape:
            loss_value = self.loss(y, self.model(X))
        return loss_value, tape.gradient(loss_value, trainable_variables)

    def one_run(self, X, y):
        return 

    def train(self, X, y):
        train_losses = []
        train_accuracy = []

        print("Begin Learning...")
        
        total_loss = 0
        
        for epoch in range(self.args.epochs):
            ctime = time.time()
            
            for e, (x_batch, y_batch) in enumerate(self.data_generator):
                
                trainable_variables = self.model.trainable_variables
                grad_accumulation = [tf.zeros_like(trainable_variables_i) \
                                    for trainable_variables_i in trainable_variables]

                loss_value, grad = self.grad(x_batch, y_batch, trainable_variables)
                total_loss += loss_value / self.args.gradient_accumulation_steps

                grad_accumulation = [(grad_accumulation+grad) for grad_accumulation, grad \
                                    in zip(grad_accumulation,grad)]

                if e % self.args.gradient_accumulation_steps:
                    grad_accumulation = [grad_i / self.args.gradient_accumulation_steps for grad_i in grad_accumulation]
                    self.optimizer.apply_gradients(zip(grad_accumulation, trainable_variables))
                    train_losses.append(total_loss)
                    total_loss = 0
            print(f"Epoch {epoch+1} \t loss value : {total_loss/self.args.gradient_accumulation_steps}")
            print(f"Time taken for epoch {epoch+1} : {round((time.time()-ctime)/60,2)} Seconds")
            
        return self.model, train_losses, train_accuracy
        