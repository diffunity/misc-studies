import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class ann():
    def __init__(self, X, y, init=None):
        self.X = X
        self.y = y
        self.classes = len(set(self.y))
        
        if init is None:

            # weight initialization (random initialization from normal distribution)
            self.W = self.initialize(self.X.shape[1],self.classes)
            self.b = np.zeros((1,self.classes))

        elif init == "xavier":

            # xavier initialization
            self.W = self.xavier_init(X.shape[1], self.classes)
            self.b = np.zeros((1, self.classes))


        
    def regularizer(self,A):
        # l2 regularization
        return self.reg*A                    
    
    def xavier_init(self, in_, out_):
        return np.random.uniform(-1,1,(in_, out_)) * math.sqrt(6./(in_+out_))    
    def initialize(self, in_, out_):
        return np.random.normal(size=(in_,out_))
    
    def activation(self, A):
        return A                             #지우고 작성하세요
    
    def softmax(self, A):
        return np.exp(A) / np.exp(A).sum()
    
    def feedForward(self, X):
        return np.apply_along_axis(self.softmax, -1, self.activation(np.matmul(X, self.W) + self.b))
    
    def train(self, lr, iteration, batch_size, reg=0.01):
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.loss_hist = []
        self.W_hist = []
        self.b_hist = []
        
        for epoch in range(iteration):
            print(f"Epoch {epoch + 1}")
            loss, W, b = self.gd()
            print(f"Loss : {loss[-1]}")
            self.loss_hist.extend(loss)
            self.W_hist.extend(W)
            self.b_hist.extend(b)
            
        self.param = (self.W_hist[-1], self.b_hist[-1])
        self.losses = (self.loss_hist[-1])
        return self.param, self.losses
    
    def gd(self):
        loss_list = []
        W_list = []
        b_list = []
        
        x_batches = np.split(self.X, len(self.X)/self.batch_size)
        y_batches = np.split(self.y, len(self.X)/self.batch_size)
        
        for inp, tar in zip(x_batches,y_batches):
            y_hat = self.feedForward(inp)
            self.loss = (-np.log(y_hat[np.eye(10)[tar].astype(bool)])).mean()
            loss_list.append(self.loss)
            
            # softmax classifier의 gradient 함수
            self.gradient = (y_hat - np.eye(10)[tar]) / tar.shape[0]
            
            # backpropagation for Weights
            dW = np.dot(inp.T, self.gradient) + self.regularizer(self.W)
            self.W -= self.lr*(dW)
            W_list.append(self.W)
            
            # backpropagation for biases
            db = self.gradient.sum(axis=0, keepdims=True)
            self.b -= self.lr*(db)
            b_list.append(self.b)
            
        return loss_list, W_list, b_list
    
    def evaluate(self, X):
        return self.feedForward(X)
    
    def get_params(self):
        return self.param



if __name__=="__main__":
    
    # data
    (train_X, train_y) , (test_X, test_y) = tf.keras.datasets.fashion_mnist.load_data()

    # MLP를 위한 shape 조정
    train_X, test_X = train_X.reshape((-1, 28*28)), test_X.reshape((-1,28*28))

    # 원활한 학습을 위한 scaling
    train_X = MinMaxScaler().fit_transform(train_X)

    train_X.shape, test_X.shape

    model = ann(train_X, train_y)

    (W, b), loss = model.train(lr=0.01, iteration=5, batch_size=60)
    plt.plot(model.loss_hist)
    plt.savefig("training_result.jpg")
    plt.clf()