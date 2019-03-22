from model import cyclegan
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser(description='')

# Directory arguments
parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='directory where logs are stored')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='summer2winter_yosemite', help='location of dataset to train on')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoints', help='location of checkpoints')

# Model descriptors
parser.add_argument('--im_size', dest='im_size', type=int, default=256, help='size of image fed into network')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='number of generator filters in first conv2d layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='number of discriminator filters in first conv2d layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='number of channels in input')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='number of channels in output')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

# Hyperparameters
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=200, help='number of epochs to run')
parser.add_argument('--epoch_decay', dest='epoch_decay', type=int, default=100, help='number of epochs to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--train_size', dest='train_size', type=int, default=1600, help='number of training samples to use')
parser.add_argument('--lambda', dest='lambda1', type=float, default=10, help='weight factor for cycle loss')
parser.add_argument('--max_pool', dest='max_pool', type=int, default=50, help='max pooling size')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        model.train(args) if args.phase == 'train' else model.test(args)

if __name__ == '__main__':
    tf.app.run()
