import numpy as np
from tensorlayer.prepro import crop
from matplotlib.pyplot import imshow, imread

from glob import glob
import random

from skimage.transform import rescale

class data_preprocessing():
    def __init__(self, path, shape, num_channels = 3, factor = 4, images = None):
        self.shape = np.array(shape)
        self.channels = num_channels
        self.factor = factor
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
        factor = self.factor
        HR_size = np.insert(np.append(self.shape, self.channels),0,batchsize)
        LR_size = np.insert(np.append(self.shape//factor, self.channels),0,batchsize)
        HR_batch = np.zeros(HR_size)
        LR_batch = np.zeros(LR_size)
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
            LR_batch[i,:,:,:] = rescale(HR_crop, .25, mode = 'constant')

        HR_batch = 2*(HR_batch.astype(float)/255.) - 1
        LR_batch = 2*(LR_batch.astype(float)/255.) - 1
        
        return HR_batch, LR_batch

def image_SR(n_images, path):
    LR_dim = 32
    paths = glob(path + '/*.jpg')
    images_paths = random.sample(paths, n_images)
    images = []
    for i in range(n_images):
        image_path = np.random.choice(images_paths)
        image = imread(image_path)
        image = 2*(image.astype(float)/255.) - 1
        
        image_shape = image.shape
        Nx,Ny = image_shape[0] // LR_dim, image_shape[1] // LR_dim
        LR_patches = np.zeros((Ny, Nx*LR_dim, LR_dim, 3))
        upscaled_image = np.zeros((Nx*LR_dim*4, Ny*LR_dim*4,3))
        for y in range(Ny):
            LR_patches[y,:,:,:] = image[y,:Nx*LR_dim,y*LR_dim:(y+1)*LR_dim,:]
            imgs= sess.run([generator.outputs], feed_dict={LR_images : LR_patches})
            upscaled_image[:,y*LR_dim*4:(y+1)*LR_dim*4,:] = imgs[y,:,:,:].squeeze()
        images.append(upscaled_image)
    return images
