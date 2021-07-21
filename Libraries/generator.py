import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tsfeatures.Util import Norm01

'''Генератор искусственных рядов'''
class generator:
    def __init__(self, steps, fake):
        self.length=len(fake)
        self.steps=steps
        self.counter=0
        self.fake=fake
        self.opti = tf.keras.optimizers.Adam(1e-4)
        self.dates=pd.date_range(start='2016-01-01', periods=self.length, freq='M').strftime('%Y-%m')
        tf.keras.backend.clear_session()
        self.model=self.make_model()
        self.x = tf.random.normal([1,self.length])
        self.x = self.model(self.x, training=False)
    def __iter__(self):
        return self
    def __next__(self):
        if self.counter < self.steps:
            with tf.GradientTape(persistent=True) as g:
                g.watch(self.model.trainable_variables)
                self.x=self.model(self.x, training=True)
                l=self.loss(self.x)
            grad = g.gradient(l, self.model.trainable_variables)
            del g
            self.opti.apply_gradients(zip(grad, self.model.trainable_variables))
            xx=Norm01(self.x.numpy().reshape(self.length))[0]
            self.counter+=1
            return xx
        else:
            raise StopIteration
    def make_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(self.length, activation='tanh',use_bias=False))
        self.model.add(layers.Dense(self.length, activation='relu',use_bias=False))
        self.model.add(layers.Dense(self.length, activation='linear'))
        return self.model
    def loss(self, x):
        return sum((x-self.fake)**2)