import numpy as np
from tensorlayer.prepro import crop
import tensorflow as tf
from matplotlib.pyplot import imshow, imread

from glob import glob
import random

#from keras.applications import vgg19
#import gc


class data_preprocessing():
    def __init__(self, path, shape, num_channels = 3, images = None):
        self.shape = np.array(shape)
        self.channels = num_channels
        self.path = path
        self.glob = None
        print('image directory: '+path)
    
    def batch(self, batchsize=32, single_image = False):
        if self.glob is None:
            self.glob = glob(self.path + '/*.jpg')
        images_paths = random.sample(self.glob, 2*batchsize)
    #    get 2*batchsize images to make sure there are at least batchsize that have the right format
        images = []
        for image_path in images_paths:
            image = imread(image_path)
            if len(image.shape)==3 and np.min(image.shape[:2])>np.min(self.shape):
                images.append(image)
        HR_size = np.insert(np.append(self.shape, self.channels),0,batchsize)
        HR_batch = np.zeros(HR_size)
        if single_image == False:
            image_set = random.sample(images, batchsize)
        else:
            image = np.random.choice(images)
        for i in range(batchsize):
            if single_image == False:
                HR_crop = crop(image_set[i].astype(float), self.shape[0], self.shape[1], is_random=True)
            else:
                HR_crop = crop(image.astype(float), self.shape[0], self.shape[1], is_random=True)
            HR_batch[i,:,:,:] = HR_crop
        HR_batch = 2*(HR_batch.astype(float)/255.) - 1
        
        return HR_batch

#class vgg_net():
#    def __init__(self, shape):
#        vgg_model = vgg19.VGG19(include_top=False,input_shape=shape)
#        self.layer_dict = {}
#        self.layers = [layer.name for layer in vgg_model.layers]
#        for layer in vgg_model.layers:
#            layer.trainable=False
#            if 'conv' in layer.name:
#                name = layer.name+'_kernel'
#                weights = layer.get_weights()[0]
#                init = tf.constant_initializer(weights)
#                self.layer_dict[name] = tf.get_variable(name, shape = weights.shape, initializer = init, trainable=False)
#                
#                name = layer.name+'_bias'
#                weights = layer.get_weights()[1]
#                init = tf.constant_initializer(weights)
#                self.layer_dict[name] = tf.get_variable(name, shape = weights.shape, initializer = init, trainable=False)
#        del vgg_model
#        gc.collect() # I got memory issues, this helps take care of it
#    
#    def conv(self,x, weights, bias):
#        y = tf.nn.conv2d(x, filter = weights, strides = [1,1,1,1],padding = 'SAME')
#        features = tf.nn.bias_add(y, bias)
#        outputs = tf.nn.relu(y)
#        return outputs
#        
#    def features(self,x):
#        y = 255*(x+tf.ones_like(x))/2.
#        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=y)
#        y = tf.concat(axis=3, values=[blue - 103.939,green - 116.779,red - 123.68])
#        outputs = {}
#        for layer in self.layers:
#            if 'conv' in layer:
#                y = self.conv(y,self.layer_dict[layer+'_kernel'],self.layer_dict[layer+'_bias'])
#                outputs[layer]=y/12.5
#            if 'pool' in layer:
#                y = tf.nn.max_pool(y, ksize=[1,2, 2,1], strides=[1,2,2,1], padding='SAME')
#        return outputs
#def perception_loss_func(features_x, features_y):
#    features = list(zip(features_x, features_y))
#    loss = 0
#    for fx,fy in features:
#        loss += tf.losses.mean_squared_error(fx, fy)
#    return loss/len(features)

