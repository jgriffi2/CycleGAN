import tensorflow as tf
from ops import *
from utils import *

def discriminator(input, options, reuse=False, name='discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        # input: H x W x C

        c1 = lrelu(instance_norm(conv2d(input, options.df_dim, ks=3, s=2, name='c1'), name='i1'), name='lr1')
        # c1: H/2 x W/2 x options.df_dim
        c2 = lrelu(instance_norm(conv2d(c1, options.df_dim*2, ks=3, s=2, name='c2'), name='i2'), name='lr2')
        # c2: H/4 x W/4 x options.df_dim*2
        c3 = lrelu(instance_norm(conv2d(c2, options.df_dim*4, ks=3, s=2, name='c3'), name='i3'), name='lr3')
        # c3: H/8 x W/8 x options.df_dim*4
        c4 = lrelu(instance_norm(conv2d(c3, options.df_dim*8, ks=3, s=1, name='c4'), name='i4'), name='lr4')
        # c4: H/8 x W/8 x options.df_dim*8
        c5 = conv2d(c4, 1, ks=3, s=1, name='c5')
        # c5: H/8 x W/8 x 1

        return c4

"""Generator"""
def generator(input, options, reuse=False, name='generator'):
    def res_block(in, nk, ks=3, s=1, name='res_block'):
        c1 = relu(instance_norm(conv2d(in, nk, ks, s, name=name+'_c1'), name=name+'_i1'), name=name+'_r1')
        c2 = instance_norm(conv2d(c1, nk, ks, s, name=name+'_c2'), name=name+'_i2')
        return in + c2

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        # input: H x W x C

        c1 = relu(instance_norm(conv2d(input, options.gf_dim, ks=9, s=1, name='c1'), name='i1'), name='r1')
        # c1: H x W x options.gf_dim
        c2 = relu(instance_norm(conv2d(c1, options.gf_dim*2, ks=3, s=2, name='c2'), name='i2'), name='r2')
        # c2: H/2 x W/2 x options.gf_dim*2
        c3 = relu(instance_norm(conv2d(c3, options.gf_dim*4, ks=3, s=2, name='c3'), name='i3'), name='r3')
        # c3: H/4 x W/4 x options.gf_dim*4

        numResBlocks = 6 if options.im_size < 256 else 9
        res = c3
        for i in range(numResBlocks):
            res = res_block(res, options.gf_dim*4, name='res' + str(i+1))
        # res: H/4 x W/4 x options.gf_dim*4

        t1 = relu(instance_norm(conv2d_transpose(res, options.gf_dim*2, ks=3, s=2, name='t1'), name='i4'), name='r4')
        # t1: H/2 x W/2 x options.gf_dim*2
        t2 = relu(instance_norm(conv2d_transpose(t1, options.gf_dim, ks=3, s=2, name='t2'), name='i5'), name='r5')
        # t2: H/2 x W/2 x options.gf_dim

        c4 = tanh(conv2d(t2, options.output_nc, ks=9, s=1, name='c4'), name='tan1')
        # c4: H x W x options.output_nc

        return c4

"""Losses"""
def gan_loss(logits, labels):
    return -tf.reduce_mean(labels * tf.log(logits)) - tf.reduce_mean((1 - labels) * tf.log(1 - logits))

def lsgan_loss(logits, labels):
    return -tf.reduce_mean(labels * tf.square(logits - 1)) + tf.reduce_mean((1 - labels) * tf.square(logits))

def wgan_loss(logits, labels):
    return -tf.reduce_mean(labels * logits) + tf.reduce_mean((1 - labels) * logits)

def cyc_loss(gen, real):
    return tf.reduce_mean(tf.abs(gen - real))
