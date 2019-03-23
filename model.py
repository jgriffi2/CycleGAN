import numpy as np
import tensorflow as tf
from collections import namedtuple
from module import *
from glob import glob
import time
from utils import ImagePool
from utils import *
import os

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess

        self.im_size = args.im_size
        self.input_nc = args.input_nc
        self.output_nc = args.output_nc
        self.lambda1 = args.lambda1
        self.dataset_dir = args.dataset_dir

        self.generator = generator
        self.discriminator = discriminator

        self.gan_loss = gan_loss
        self.cyc_loss = cyc_loss

        OPTIONS = namedtuple('OPTIONS', 'gf_dim df_dim output_nc im_size')
        self.options = OPTIONS._make((args.ngf, args.ndf, args.output_nc, args.im_size))

        self.pool = ImagePool(args.max_pool)

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.real_X = tf.placeholder(tf.float32,
                                     [None, self.im_size, self.im_size, self.input_nc],
                                     name='real_X')
        self.real_Y = tf.placeholder(tf.float32,
                                     [None, self.im_size, self.im_size, self.output_nc],
                                     name='real_Y')

        self.fake_X = self.generator(self.real_Y, self.options, reuse=False, name='genYtoX')
        self.fake_Y = self.generator(self.real_X, self.options, reuse=False, name='genXtoY')

        self.fake_X_ = self.generator(self.fake_Y, self.options, reuse=True, name='genYtoX')
        self.fake_Y_ = self.generator(self.fake_X, self.options, reuse=True, name='genXtoY')

        self.d_fake_X = self.discriminator(self.fake_X, self.options, reuse=False, name='disX')
        self.d_fake_Y = self.discriminator(self.fake_Y, self.options, reuse=False, name='disY')

        self.g_G_loss = self.gan_loss(self.d_fake_Y, tf.ones_like(self.d_fake_Y))
        self.g_F_loss = self.gan_loss(self.d_fake_X, tf.ones_like(self.d_fake_X))
        self.g_c_loss = self.cyc_loss(self.fake_X_, self.real_X) \
                    + self.cyc_loss(self.fake_Y_, self.real_Y)
        self.g_loss = self.g_G_loss + self.g_F_loss + self.lambda1 * self.g_c_loss

        self.fake_X_pool = tf.placeholder(tf.float32,
                                          [None, self.im_size, self.im_size, self.input_nc],
                                          name='fake_X_pool')
        self.fake_Y_pool = tf.placeholder(tf.float32,
                                          [None, self.im_size, self.im_size, self.output_nc],
                                          name='fake_Y_pool')

        self.d_real_X = self.discriminator(self.real_X, self.options, reuse=True, name='disX')
        self.d_real_Y = self.discriminator(self.real_Y, self.options, reuse=True, name='disY')

        self.d_fake_X_pool = self.discriminator(self.fake_X_pool, self.options, reuse=True, name='disX')
        self.d_fake_Y_pool = self.discriminator(self.fake_Y_pool, self.options, reuse=True, name='disY')

        self.d_X_loss = self.gan_loss(self.d_real_X, tf.ones_like(self.d_real_X)) \
                      + self.gan_loss(self.d_fake_X_pool, tf.zeros_like(self.d_fake_X_pool))
        self.d_Y_loss = self.gan_loss(self.d_real_Y, tf.ones_like(self.d_real_Y)) \
                      + self.gan_loss(self.d_fake_Y_pool, tf.zeros_like(self.d_fake_Y_pool))
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

        self.test_X = tf.placeholder(tf.float32,
                                     [None, self.im_size, self.im_size, self.input_nc],
                                     name='test_X')
        self.test_Y = tf.placeholder(tf.float32,
                                     [None, self.im_size, self.im_size, self.output_nc],
                                     name='test_Y')

        self.fake_test_X = self.generator(self.test_Y, self.options, reuse=True, name='genYtoX')
        self.fake_test_Y = self.generator(self.test_X, self.options, reuse=True, name='genXtoY')

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, name='dis_optim') \
                                              .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, name='gen_optim') \
                                              .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

        dataX = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainX'))
        dataY = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainY'))

        self.num_batches = min(min(len(dataX), len(dataY)), args.train_size) // args.batch_size

        iter = 1
        start_time = time.time()
        for epoch in range(args.num_epochs):
            np.random.shuffle(dataX)
            np.random.shuffle(dataY)

            lr = args.lr if epoch < args.epoch_decay else args.lr * (args.num_epochs - epoch) / (args.num_epochs - args.epoch_decay)

            for batch_num in range(self.num_batches):
                batch_filesX = dataX[batch_num * args.batch_size : (batch_num+1) * args.batch_size]
                batch_filesY = dataY[batch_num * args.batch_size : (batch_num+1) * args.batch_size]

                batch_imagesX = load_data(batch_filesX, self.im_size)
                batch_imagesY = load_data(batch_filesY, self.im_size)

                _, summary_str, fake_X, fake_Y = self.sess.run(
                    [self.g_optim, self.g_loss_sum, self.fake_X, self.fake_Y],
                    feed_dict={self.real_X: batch_imagesX, self.real_Y: batch_imagesY,
                               self.lr: lr})
                [fake_X, fake_Y] = self.pool([fake_X, fake_Y])
                self.writer.add_summary(summary_str, iter)

                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_loss_sum],
                    feed_dict={self.real_X: batch_imagesX, self.real_Y: batch_imagesY,
                               self.fake_X_pool: fake_X, self.fake_Y_pool: fake_Y,
                               self.lr: lr})
                self.writer.add_summary(summary_str, iter)

                if np.mod(iter, args.save_freq) == 0:
                    self.save(args.checkpoint_dir, iter)

                print("Epoch: [%d] [%d/%d] time: %.4f" % (epoch, batch_num, self.num_batches, time.time() - start_time))

                iter += 1

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if not self.load(args.checkpoint_dir):
            return

        filesX = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testX'))
        filesY = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testY'))

        imagesX = load_data(filesX, self.im_size, phase=args.phase)
        imagesY = load_data(filesY, self.im_size, phase=args.phase)

        fakeX, fakeY = self.sess.run([self.fake_test_X, self.fake_test_Y],
                                     feed_dict={self.test_X: imagesX, self.test_Y: imagesY})

        for i in range(len(filesX)):
            print('/XtoY_' + os.path.basename(filesX[i]))
            imsave(args.test_dir + '/XtoY_' + os.path.basename(filesX[i]), fakeY[i])

        for i in range(len(filesY)):
            print('/YtoX_' + os.path.basename(filesY[i]))
            imsave(args.test_dir + '/YtoX_' + os.path.basename(filesY[i]), fakeX[i])

    def load(self, checkpoint_dir):
        print('[*] Reading checkpoint...')

        model_dir = "%s_%s" % (self.dataset_dir, self.im_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, checkpoint_name))
            print('[*] Load success')
            return True

        print('[!] Load failed')
        return False

    def save(self, checkpoint_dir, iter):
        model_name = 'cyclegan.model'
        model_dir = "%s_%s" % (self.dataset_dir, self.im_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=iter)
