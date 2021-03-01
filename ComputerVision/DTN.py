import tensorflow as tf

class F(tf.keras.layers.Layer):

    def __init__(self):
        super(F, self).__init__()
    
    def forward(self, X):
        net = tf.keras.layers.Conv2D(64, [3,3])(X)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(128, [3,3])(X)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(256, [3,3])(X)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(128, [4,4])(X)
        net = tf.keras.layers.BatchNormalization()(net)
        

class DTN(tf.keras.Model):

    def __init__(self):
        super(DTN, self).__init__()