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
flags.DEFINE_integer("LR_size", 32, "The size of low resolution patch to use [32]")
flags.DEFINE_integer("HR_size", 128, "The size of the high resolution patch to produce [128]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("checkpoint_dir", None, "checkpoints directory path [None]")
flags.DEFINE_string("sample_path", "./upscaled_image", "path to save the upscaled image [./upscaled_image]")
flags.DEFINE_string("image_path", "./image", "path to the image to be upscaled [./image]")
FLAGS = flags.FLAGS

def image_SR(image, batch_size, sess, generator, NU, LR_images):
#    generates both the image reconstructed by the network and the one standard upscaling would have returned.
    
#    I could divide the image in patches of size 32x32, but there is a chance doing so would create visible lines between each patch. So instead, I use overlapping patches.
#    The parameter k controls the amount of overlap.
    k=2
    LR_dim = 32
    image_shape  = image.shape
    
    print('processing image')
    Nx,Ny = image_shape[0] // LR_dim, image_shape[1] // LR_dim
    LR_patch = np.zeros((1, LR_dim, LR_dim, 3))
    upscaled_image = np.zeros((Nx*LR_dim*4, Ny*LR_dim*4,3))
    naive_upscaling = np.zeros((Nx*LR_dim*4, Ny*LR_dim*4,3))
    scale = np.zeros((Nx*LR_dim*4, Ny*LR_dim*4,3))
    for y in range(k*Ny-1):
        for x in range(k*Nx-1):
            LR_patch[0,:,:,:] = image[x*LR_dim//k:(x+k)*LR_dim//k,y*LR_dim//k:(y+k)*LR_dim//k,:]
            img, nimg = sess.run([generator.outputs, NU.outputs], feed_dict={LR_images : LR_patch})
            upscaled_image[(x*LR_dim*4)//k:((x+k)*LR_dim*4)//k,(y*LR_dim*4)//k:((y+k)*LR_dim*4)//k,:] += img[0,:,:,:].squeeze()
            naive_upscaling[(x*LR_dim*4)//k:((x+k)*LR_dim*4)//k,(y*LR_dim*4)//k:((y+k)*LR_dim*4)//k,::] += nimg[0,:,:,:].squeeze()
            scale[(x*LR_dim*4)//k:((x+k)*LR_dim*4)//k,(y*LR_dim*4)//k:((y+k)*LR_dim*4)//k,:] += 1
    upscaled_image = upscaled_image/scale
    naive_upscaling = naive_upscaling/scale
    print('processing finished')
    return upscaled_image, naive_upscaling

def main(_):
    
    pp.pprint(flags.FLAGS.__flags)
    
    path = FLAGS.data_dir
    batch_size = int(np.max(shapes[:,1]))//FLAGS.LR_size
    
    with tf.device("/gpu:0"):
        LR_images =  tf.placeholder(tf.float32, [None, FLAGS.LR_size, FLAGS.LR_size, FLAGS.c_dim], name='LR_images')
        generator, naive_upscaling = model.generator(LR_images, is_train=False, reuse=False)
   
    g_vars = tl.layers.get_variables_with_name('generator', True, False)
    saver = tf.train.Saver(g_vars)
    
    image_path = FLAGS.image_path
    
    image = imread(images_path)
    image = 2*(image.astype(float)/255.) - 1
    
    with tf.Session() as sess: 
        tl.layers.initialize_global_variables(sess)    
        # load the latest checkpoints
        if tl.files.file_exists(checkpoint_dir+'checkpoint'):
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
        print('generator model loaded')
            
        path = FLAGS.data_dir
        upscaled_image, naive_upscaled = image_SR(image_list, batch_size, sess, generator, naive_upscaling, LR_images)
            tl.visualize.save_image(upscaled_images, './images/upscaled'+str(i)+'.png')
            tl.visualize.save_image(naive_upscaled, './images/naive'+str(i)+'.png')
            tl.visualize.save_image(image_list, './images/original'+str(i)+'.png')

if __name__ == '__main__':
    tf.app.run()

