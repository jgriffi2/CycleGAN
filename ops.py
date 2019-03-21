import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def conv2d(input, nk, ks=3, s=1, padding='same', name='conv2d'):
    return layers.conv2d(input, nk, ks, s, padding, activation_fn=None, scope=name)

def conv2d_transpose(input, nk, ks=3, s=1, padding='same', name='conv2d'):
    return layers.conv2d_transpose(input, nk, ks, s, padding, activation_fn=None, scope=name)

"""Normalizations"""
def batch_norm(input, name='batch_norm'):
    return layers.batch_norm(input, scope=name)

def instance_norm(input, name='instance_norm'):
    return layers.instance_norm(input, scope=name)

"""Activations"""
def relu(input, name='relu'):
    return tf.nn.relu(input, name)

def lrelu(input, alpha=0.2, name='lrelu'):
    return tf.nn.leaky_relu(input, alpha, name)

def tanh(input, name='tanh'):
    return tf.nn.tanh(input, name)
