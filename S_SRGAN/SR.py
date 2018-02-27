import os, sys, pprint, time
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
import model
from utils import *



pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("checkpoint_dir", None, "checkpoints directory path [None]")
flags.DEFINE_string("sample_path", "./upscaled_image", "path to save the upscaled image [./upscaled_image]")
flags.DEFINE_string("image_path", "./image", "path to the image to be upscaled [./image]")
FLAGS = flags.FLAGS

#example use: 

#python SR.py --image_path='./image.jpg' -checkpoint_dir='./runs/run_0/checkpoints' --sample_path='./result'

def main(_):
    
    pp.pprint(flags.FLAGS.__flags)
    checkpoint_dir = FLAGS.checkpoint_dir
    
    if checkpoint_dir is None:
        print('please specify a checkpoint directory')
        return
    
    c_dim = FLAGS.c_dim
    
    with tf.device("/gpu:0"):
#        LR_image =  tf.placeholder(tf.float32, [1, LR_size, LR_size, c_dim], name='LR_image')
        LR_image =  tf.placeholder(tf.float32, [1, None, None, c_dim], name='LR_image')
        # generator for training
        generator = model.generator(LR_image, is_train=True, reuse=False)
        generated_image = generator.outputs
   
    g_vars = tl.layers.get_variables_with_name('generator', True, False)
    saver = tf.train.Saver(g_vars)
    
    image_path = FLAGS.image_path
    image = imread(image_path)
    if image.max()>254:
        image = image.astype(float)/255.
    image = 2*image - 1
        
    sample_path = FLAGS.sample_path
    
    with tf.Session() as sess: 
        tl.layers.initialize_global_variables(sess)    
        # load the latest checkpoints
        if tl.files.file_exists(checkpoint_dir+'/checkpoint'):
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
        print('generator model loaded')    
#        upscaled_image = proc.SR(image, sess, generator, LR_image)
        upscaled_image = sess.run(tf.squeeze(generated_image), feed_dict = {LR_image : np.expand_dims(image, axis=0)})
        upscaled_image = np.clip(upscaled_image, -1.,1.)
        upscaled_image = (upscaled_image+1)/2.
        tl.visualize.save_image(upscaled_image, sample_path+'.png')
        

if __name__ == '__main__':
    tf.app.run()

