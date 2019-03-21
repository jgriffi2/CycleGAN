import numpy as np
import tensorflow as tf
from collections import namedtuple
from module import *

class cyclegan(object):
    def __init__(self, args):
        self.im_size = args.im_size
        self.lambda = args.lambda

        self.generator = generator
        self.discriminator = discriminator

        self.gan_loss = gan_loss
        self.cyc_loss = cyc_loss

        OPTIONS = namedtuple('OPTIONS', 'gf_dim df_dim output_nc')
        self.options = OPTIONS.make((args.ngf, args.ndf, args.output_nc))

    def _build_model(self):
        self.real_X = tf.placeholder(tf.float32,
                                     [None, self.im_size, self.im_size, self.input_nc],
                                     name='real_X')
        self.real_Y = tf.placeholder(tf.float32,
                                     [None, self.im_size, self.im_size, self.output_nc],
                                     name='real_Y')

        self.fake_X = self.generator(real_Y, self.options, reuse=False, name='genYtoX')
        self.fake_Y = self.generator(real_X, self.options, reuse=False, name='genXtoY')

        self.fake_X_ = self.generator(fake_Y, self.options, reuse=True, name='genYtoX')
        self.fake_Y_ = self.generator(fake_X, self.options, reuse=True, name='genXtoY')

        self.d_real_X = self.discriminator(self.real_X, self.options, reuse=False, name='disX')
        self.d_real_Y = self.discriminator(self.real_Y, self.options, reuse=False, name='disY')

        self.d_fake_X = self.discriminator(self.fake_X, self.options, reuse=True, name='disX')
        self.d_fake_Y = self.discriminator(self.fake_Y, self.options, reuse=True, name='disY')

        self.g_G_loss = self.gan_loss(self.d_fake_Y, tf.ones_like(self.d_fake_Y))
        self.g_F_loss = self.gan_loss(self.d_fake_X, tf.ones_like(self.d_fake_X))
        self.g_c_loss = self.cyc_loss(self.fake_X_, self.real_X) \
                    + self.cyc_loss(self.fake_Y_, self.real_Y)
        self.g_loss = self.g_G_loss + self.g_F_loss + self.lambda * self.g_c_loss

        self.d_X_loss = self.gan_loss(self.d_real_X, tf.ones_like(self.d_real_X)) \
                      + self.gan_loss(self.d_fake_X, tf.zeros_like(self.d_fake_X))
        self.d_Y_loss = self.gan_loss(self.d_real_Y, tf.ones_like(self.d_real_Y)) \
                      + self.gan_loss(self.d_fake_Y, tf.zeros_like(self.d_fake_Y))
        self.d_loss = self.d_X_loss + self.d_Y_loss

        self.loss = self.g_loss + self.d_loss

        self.g_G_loss_sum = tf.summary.scalar('g_G_loss', self.g_G_loss)
        self.g_F_loss_sum = tf.summary.scalar('g_F_loss', self.g_F_loss)
        self.g_c_loss_sum = tf.summary.scalar('g_c_loss', self.g_c_loss)
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)

        self.d_X_loss_sum = tf.summary.scalar('d_X_loss', self.d_X_loss)
        self.d_Y_loss_sum = tf.summary.scalar('d_Y_loss', self.d_Y_loss)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

        self.loss_sum = tf.summary.scalar('loss', self.loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        for var in t_vars: print(var.name)
