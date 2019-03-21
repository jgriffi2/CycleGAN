import cyclegan
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--im_size', dest='im_size', type=int, default=256, help='size of image fed into network')
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='number of generator filters in first conv2d layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=32, help='number of discriminator filters in first conv2d layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='number of channels in input')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='number of channels in output')

if __name__ == '__main__':
    tf.app.run()
