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
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("data_dir", "../../datasets/birds", "Directory name to save the image samples [../../datasets/birds]")
FLAGS = flags.FLAGS

def image_choice(n_images, path):
    LR_dim = 32
    paths = glob(path + '/*.jpg')
    images_paths = random.sample(paths, n_images)
    
    images = []
    shapes = np.zeros([n_images, 3])
    for i in range(n_images):
        image = imread(images_paths[i])
        image = 2*(image.astype(float)/255.) - 1
        images.append(image)
        shapes[i,:] = image.shape
    return images, shapes
    
def image_SR(images, batch_size, sess, generator, NU, LR_images):
    LR_dim = 32
    upscaled_images = []
    naive_upscaled = []
    index = 0
    for image in images:
        print('processing image '+str(index)+' of '+str(len(images)))
        index+=1
        image_shape  = image.shape
        Nx,Ny = image_shape[0] // LR_dim, image_shape[1] // LR_dim
        LR_patch = np.zeros((1, LR_dim, LR_dim, 3))
        upscaled_image = np.zeros((Nx*LR_dim*4, Ny*LR_dim*4,3))
        naive_upscaling = np.zeros((Nx*LR_dim*4, Ny*LR_dim*4,3))
        for y in range(2*Ny-1):
            for x in range(2*Nx-1):
                LR_patch[0,:,:,:] = image[x*LR_dim//2:(x+2)*LR_dim//2,y*LR_dim//2:(y+2)*LR_dim//2,:]
                img, nimg = sess.run([generator.outputs, NU.outputs], feed_dict={LR_images : LR_patch})
                upscaled_image[x*LR_dim*2:(x+2)*LR_dim*2,y*LR_dim*2:(y+2)*LR_dim*2,:] += img[0,:,:,:].squeeze()
                naive_upscaling[x*LR_dim*2:(x+2)*LR_dim*2,y*LR_dim*2:(y+2)*LR_dim*2,:] += nimg[0,:,:,:].squeeze()
        upscaled_image[1*LR_dim*2:(2*Nx-1)*LR_dim*2,:,:]/=2
        upscaled_image[:,1*LR_dim*2:(2*Ny-1)*LR_dim*2,:]/=2
        upscaled_images.append(upscaled_image)
        naive_upscaling[1*LR_dim*2:(2*Nx-1)*LR_dim*2,:,:]/=2
        naive_upscaling[:,1*LR_dim*2:(2*Ny-1)*LR_dim*2,:]/=2
        naive_upscaled.append(naive_upscaling)
    print('processing finished')
    return upscaled_images, naive_upscaled

def main(_):
    
    pp.pprint(flags.FLAGS.__flags)
    
    path = FLAGS.data_dir
    image_list, shapes = image_choice(6, path)
    batch_size = int(np.max(shapes[:,1]))//FLAGS.LR_size
    
    with tf.device("/gpu:0"):
        LR_images =  tf.placeholder(tf.float32, [None, FLAGS.LR_size, FLAGS.LR_size, FLAGS.c_dim], name='LR_images')
        generator, naive_upscaling = model.generator(LR_images, is_train=False, reuse=False)
   
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    
    model_dir = "%s_%s" % (32, FLAGS.LR_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    
    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    if tl.files.file_exists(net_g_name):
        load_params_generator = tl.files.load_npz(name=net_g_name)
        tl.files.assign_params(sess, load_params_generator, generator)
        print('generator model loaded')
        
    path = FLAGS.data_dir
    upscaled_images, naive_upscaled = image_SR(image_list, batch_size, sess, generator, naive_upscaling, LR_images)
    original_images = image_list
    for i in range(6):
        tl.visualize.save_image(upscaled_images[i], './images/upscaled'+str(i)+'.png')
        tl.visualize.save_image(naive_upscaled[i], './images/naive'+str(i)+'.png')
        tl.visualize.save_image(image_list[i], './images/original'+str(i)+'.png')

if __name__ == '__main__':
    tf.app.run()

