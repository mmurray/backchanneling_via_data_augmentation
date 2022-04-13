import tensorflow as tf
from tensorflow import keras


class LSTM(keras.Model):

    def __init__(self, output_bias=None, dropout=0.2):
        super(LSTM, self).__init__()
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        self.lstm = keras.layers.LSTM(64, time_major=False, return_sequences=True)
        self.dropout = keras.layers.Dropout(rate=dropout)
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(1, activation=tf.nn.sigmoid, name='output', bias_initializer=output_bias)

    def call(self, x):
        x = tf.concat((x['head_pose'], x['mfcc'], x['f0']), axis=-1)
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return self.dense(x)


