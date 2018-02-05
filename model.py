import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def generator(inputs, is_train=True, reuse=False):
    B = 16
    c_dim = FLAGS.c_dim # n_color 3
#    batch_size = FLAGS.batch_size # 32
    
    w_init = tf.glorot_normal_initializer()

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
    
#    inputs have size 32x32x3

#    inputs layer
        x = InputLayer(inputs, name='inputs')
#        layer 0
        nf0 = 64
        conv0 = Conv2dLayer(layer = x, shape =[9,9,c_dim, nf0], act = tf.identity, padding='SAME', name = 'conv0')
        conv0 = PReluLayer(layer = conv0, name='prelu0')
        
        num_conv=0
        with tf.variable_scope("resloop", reuse=reuse):
            conv = conv0
            res = conv0
            for i in range(B):
                conv = Conv2dLayer(layer = conv, shape =[3,3,nf0, nf0], act = tf.identity, padding='SAME', name = 'conv'+str(num_conv))
                bn = BatchNormLayer(layer = conv, is_train=is_train, name='bn'+str(num_conv))
                bn = PReluLayer(layer = bn, name='prelu'+str(num_conv))
                num_conv+=1
                conv = Conv2dLayer(layer = bn, shape =[3,3,nf0, nf0], act = tf.nn.relu, padding='SAME', name = 'conv'+str(num_conv))
                bn = BatchNormLayer(layer = conv, is_train=is_train, name='bn'+str(num_conv))
                bn = PReluLayer(layer = bn, name='prelu'+str(num_conv))
                num_conv+=1
                conv = ElementwiseLayer([bn, res], combine_fn=tf.add, name = 'add'+str(i))
                res = conv
        conv =  Conv2dLayer(layer = conv, shape =[3,3,nf0, nf0], act = tf.identity, padding='SAME', name = 'conv1')
        bn = BatchNormLayer(layer = conv, is_train=is_train, name='bn1')
        merge = ElementwiseLayer([bn, conv0], combine_fn=tf.add, name = 'add')
        
        nf1 = 256
        conv =  Conv2dLayer(layer = merge, shape =[3,3,nf0, nf1], act = tf.identity, padding='SAME', name = 'conv2')
        SP = SubpixelConv2d(conv, scale=2, n_out_channel=None, act=tf.identity, name='subpixel_0')
        bn = PReluLayer(layer = SP, name='prelu2')
                
        nf2 = 256
        conv =  Conv2dLayer(layer = SP, shape = [3,3,nf1//4, nf2], act = tf.identity, padding='SAME', name = 'conv3')
        SP = SubpixelConv2d(conv, scale=2, n_out_channel=None, act=tf.identity, name='subpixel_1')
        bn = PReluLayer(layer = SP, name='prelu3')
        
        output =  Conv2dLayer(layer = SP, shape =[9,9,nf2//4, c_dim], act = tf.identity, padding='SAME', name = 'conv4')
                
        up_input = UpSampling2dLayer(layer = x, size=[2,2], is_scale=True, name='up_input0')
        up_input = UpSampling2dLayer(layer = up_input, size=[2,2], is_scale=True, name='up_input1')

    return output, up_input

def discriminator(inputs, is_train=True, reuse=False):
    
    c_dim = FLAGS.c_dim # n_color 3
    w_init = tf.glorot_normal_initializer()
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        lrelu = lambda x : tl.act.lrelu(x,alpha = .2)
#    inputs layer
        inputs = InputLayer(inputs, name='inputs')

        nf0 = 32
        conv = Conv2dLayer(layer = inputs, shape = [3,3, c_dim, nf0], act=lrelu, name='conv0')
        conv = Conv2dLayer(layer = conv, shape = [3,3, nf0, nf0], act=tf.identity,strides =[1,2,2,1], name='conv1')
        bn = BatchNormLayer(layer = conv, is_train=is_train, act = lrelu, name='bn0')
        
        nf1 = 2*nf0
        conv = Conv2dLayer(layer = bn, shape = [3,3, nf0, nf1], act=tf.identity, name='conv2')
        bn = BatchNormLayer(layer = conv, is_train=is_train, act = lrelu, name='bn1')
        conv = Conv2dLayer(layer = bn, shape = [3,3, nf1, nf1], act=tf.identity, strides = [1,2,2,1], name='conv3')
        bn = BatchNormLayer(layer = conv, is_train=is_train, act = lrelu, name='bn2')
        
        nf2 = 2*nf1
        conv = Conv2dLayer(layer = bn, shape = [3,3, nf1, nf2], act=tf.identity, name='conv4')
        bn = BatchNormLayer(layer = conv, is_train=is_train, act = lrelu, name='bn3')
        conv = Conv2dLayer(layer = bn, shape = [3,3, nf2, nf2], act=tf.identity, strides = [1,2,2,1], name='conv5')
        bn = BatchNormLayer(layer = conv, is_train=is_train, act = lrelu, name='bn4')

        nf3 = 2*nf2
        conv = Conv2dLayer(layer = bn, shape = [3,3, nf2, nf3], act=tf.identity, name='conv6')
        bn = BatchNormLayer(layer = conv, is_train=is_train, act = lrelu, name='bn5')
        conv = Conv2dLayer(layer = bn, shape = [3,3, nf3, nf3], act=tf.identity, strides = [1,2,2,1], name='conv7')
        bn = BatchNormLayer(layer = conv, is_train=is_train, act = lrelu, name='bn6')

        flatten = FlattenLayer(bn, name='flatten')
        dense = DenseLayer(flatten, n_units=1024, act=lrelu, name='dense0')
        output = DenseLayer(dense, n_units=1, act=tf.identity, name='dense1')
    return output
