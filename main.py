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

from keras.applications import vgg19

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("updates", 100, "Updates per epoch [100]")
flags.DEFINE_float("learning_rate", 0.001, "initial Learning rate for adam [0.1]")
flags.DEFINE_float("decay_rate", .9, "learning rate decay per thousand updates - set to 1. for constant learning rate [0.9]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 16, "The number of batch images [16]")
flags.DEFINE_integer("LR_size", 32, "The size of low resolution patch to use [32]")
flags.DEFINE_integer("HR_size", 128, "The size of the high resolution patch to produce [128]")
flags.DEFINE_integer("sample_size", 32, "The number of sample images [32]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 100, "The epoch interval of generating sample. [100]")
flags.DEFINE_integer("save_step", 50, "The interval of saving checkpoints. [50]")
flags.DEFINE_string("data_dir", "../../datasets/birds", "Directory name to save the image samples [../../datasets/birds]")
flags.DEFINE_string("run_name", "run_", "name of the run")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [True]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("load_from_checkpoint", True, "if loading from a previous training session")
flags.DEFINE_float("content_scaling", 1e2, "scaling of the content loss [1e2]")
flags.DEFINE_float("perception_scaling", 1e3, "scaling of the perceptual  loss [1e3]")
flags.DEFINE_float("discriminator_lr_scale", .1, "scaling of the generator vs discriminator learning rate [.1]")
flags.DEFINE_integer("naming_offset", 0, "to resume the naming of samples from a previous session [0]")
FLAGS = flags.FLAGS


class vgg_loss():
    def __init__(self, shape):
        self.vgg = vgg19.VGG19(include_top=False,input_shape=shape)
        for layer in self.vgg.layers:
            layer.trainable=False
    def features(self,x):
        y = 255*(x+1)/2
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=y)
        y = tf.concat(axis=3, values=[blue - 123.68,green - 116.779,red - 103.939])
        outputs = []
        for layer in self.vgg.layers:
            y = layer(y)
            if 'conv' in layer.name:
                outputs.append(y/12.75)
        return outputs
    def loss(self,features_x, features_y):
        features = list(zip(features_x, features_y))
        loss = 0
        for fx,fy in features:
            loss += tl.cost.mean_squared_error(fx, fy, is_mean=True)
        loss = loss/len(features)
        return loss

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    g_content_scaling = FLAGS.content_scaling
    g_perception_scaling = FLAGS.perception_scaling
    d_lr_scale = FLAGS.discriminator_lr_scale    
    
    ## forcing it to use the GPU only runs into memory issues on my machine    
    # with tf.device('/GPU:0'):
    ##========================= DEFINE MODEL ===========================##
    HR_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.HR_size, FLAGS.HR_size, FLAGS.c_dim], name='HR_images')
    LR_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.LR_size, FLAGS.LR_size, FLAGS.c_dim], name='LR_images')

    # generator for training
    generator, _ = model.generator(LR_images, is_train=True, reuse=False)
    # generated fake images --> discriminator
    discriminator_fake = model.discriminator(generator.outputs, is_train=True, reuse=False)
    # real images --> discriminator
    discriminator_real = model.discriminator(HR_images, is_train=True, reuse=True)
    # generator for sample generation, set is_train to False so that BatchNormLayer behave differently
    sample_gen, naive_upscaler = model.generator(LR_images, is_train=False, reuse=True)
    # the result of a basic upsampling is generated for reference
    upscaled_image = naive_upscaler.outputs

    ##========================= DEFINE TRAIN OPS =======================##
    
    dr_logits= discriminator_real.outputs
    df_logits= discriminator_fake.outputs
    
    # discriminator: real images are labelled as 1
    d_loss_real = tl.cost.sigmoid_cross_entropy(dr_logits, tf.ones_like(dr_logits), name='dreal')
    # discriminator: images from the generator (fake) are labelled as 0
    d_loss_fake = tl.cost.sigmoid_cross_entropy(df_logits, tf.zeros_like(df_logits), name='dfake')
    # total loss is the sum of the previous two
    d_loss = d_loss_real + d_loss_fake
    
    # adversarial generator loss: try to make the the fake images look real (1)
    g_loss_adversarial = tl.cost.sigmoid_cross_entropy(df_logits, tf.ones_like(df_logits), name='gfake')
    
    # builds the VGG perception loss detailed in original paper
    vgg = vgg_loss([128,128,3])
    with tf.variable_scope("perception", reuse = False):
        tl.layers.set_name_reuse(False)
        perception_input = InputLayer(HR_images, name='perception_input')
        perception_output = LambdaLayer(perception_input, fn=vgg.features, name='perception_output')       
    with tf.variable_scope("perception", reuse = True):
        tl.layers.set_name_reuse(True)
        perception_input_gen = InputLayer(generator.outputs, name='perception_input_gen')
        perception_output_gen = LambdaLayer(perception_input_gen, fn=vgg.features, name='perception_output_gen')       
    features_real = perception_output.outputs
    features_gen = perception_output_gen.outputs
    g_loss_perception = vgg.loss(features_real, features_gen)
    g_loss = g_loss_adversarial/g_perception_scaling + g_loss_perception
    
    # list of generator and discriminator variables for use by the optimizers
    g_vars = tl.layers.get_variables_with_name('generator', True, False)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, False)
    
    n_vars_d= len(d_vars)
    n_vars_g= len(g_vars)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = FLAGS.learning_rate
    decay_rate = FLAGS.decay_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,decay_steps = 2*1e3, decay_rate = decay_rate, staircase=False)

    d_optim = tf.train.AdamOptimizer(d_lr_scale*learning_rate, beta1=FLAGS.beta1).minimize(d_loss, var_list = d_vars, global_step = global_step)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta1).minimize(g_loss, var_list = g_vars,global_step = global_step)
    
    g_grads_adversarial = tf.gradients(g_loss_adversarial, g_vars)
    g_grads_perception = tf.gradients(g_loss_perception, g_vars)

    with tf.name_scope('gradient_histograms/'):
        for g0,g1, var in zip(g_grads_adversarial, g_grads_perception, g_vars):
            if not 'prelu' in var.name:
                if not (('Reshape' in g0.op.name) or ('batchnorm' in g0.op.name)):
                    tf.summary.histogram('adversarial/'+g0.op.name, g0)
                    tf.summary.histogram('content/'+g1.op.name, g1)
    
    with tf.name_scope('loss_summaries/'):
        tf.summary.scalar('generator_adversarial', g_loss_adversarial)
        tf.summary.scalar('generator_perception', g_loss_perception)
        tf.summary.scalar('discriminator_real', d_loss_real)
        tf.summary.scalar('discriminator_fake', d_loss_fake)
        tf.summary.scalar('discriminator_total', d_loss)
        tf.summary.scalar('learning_rate', learning_rate)
    
    merged = tf.summary.merge_all()    
    saver = tf.train.Saver(g_vars+d_vars)
        
    with tf.Session() as sess:
        tl.layers.initialize_global_variables(sess)    

        # generates the various folders in case they don't already exist, starting with the run directory
        run_dir = './runs/'+FLAGS.run_name+'/'    
        checkpoint_dir = run_dir+'checkpoints/'
        sample_dir = run_dir+'samples/'
        logs_dir = run_dir+'logs/'

        tl.files.exists_or_mkdir(run_dir)    
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(sample_dir)
        tl.files.exists_or_mkdir(logs_dir)
        
        #creates the tensorboard log writer
        writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

        save_path = checkpoint_dir+'/model.ckpt'
        print('the model will be saved at '+ save_path)

        # load the latest checkpoints if load_from_checkpoint is set to True. Otherwise, generates a summary of the new network
        if FLAGS.load_from_checkpoint:
            if tl.files.file_exists(checkpoint_dir+'checkpoint'):
                print('loading model from '+checkpoint_dir)
                saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
                print('model loaded')
            else:
                print('model not found, starting from scratch')
                generator.print_params()
                discriminator_fake.print_params()
        else:
            generator.print_params()
            discriminator_fake.print_params()

        path = FLAGS.data_dir   
        shape = [FLAGS.HR_size, FLAGS.HR_size]
        
        # creates the image pipeline
        data_preproc = data_preprocessing(path,shape)

        # sets the parameters for saving the sample images
        batchsize = FLAGS.batch_size
        n_display_0 = int(np.sqrt(batchsize))+1
        n_display_1 = batchsize//n_display_0+1

        ##========================= TRAIN MODELS ================================##
        naming_offset = FLAGS.naming_offset
        iter_counter = naming_offset*FLAGS.updates
        
        # saves the reference images
        HR_sample, LR_sample = data_preproc.batch(batchsize = batchsize, single_image=False)
        up_img = sess.run(upscaled_image, feed_dict={LR_images : LR_sample, HR_images: HR_sample})
        tl.visualize.save_images(up_img, [n_display_0, n_display_1+1], '{}train_LR.png'.format(sample_dir))
        tl.visualize.save_images(HR_sample, [n_display_0, n_display_1+1], '{}train_HR.png'.format(sample_dir))
        
        for epoch in range(FLAGS.epoch):
            epoch_time = time.time()
            for update in range(FLAGS.updates):
                start_time = time.time()
                
                # generate low and high-resolution image batch:
                HR_batch, LR_batch = data_preproc.batch(batchsize = batchsize, single_image=False)

                # train the network and generate summary statistics for tensorboard
                summary,_,_ = sess.run([merged,g_optim, d_optim], feed_dict={LR_images: LR_batch, HR_images: HR_batch })
                print("Epoch: [%2d/%2d] [%3d/%3d]" % (epoch+naming_offset, FLAGS.epoch+naming_offset, update, FLAGS.updates))
                writer.add_summary(summary, iter_counter)
                
                if epoch>0: # only save things if the training has actually started
                    if np.mod(iter_counter, FLAGS.sample_step) == 0:
                        # generate and visualize sample images
                        img, _ = sess.run([sample_gen.outputs, upscaled_image], feed_dict={LR_images : LR_sample, HR_images: HR_sample})
                        tl.visualize.save_images(img, [n_display_0, n_display_1+1], '{}train_{:02d}.png'.format(sample_dir, epoch+naming_offset))
                        
                    if np.mod(iter_counter, FLAGS.save_step) == 0:
                        # save current network parameters
                        saver.save(sess, save_path, global_step = (epoch+naming_offset)*FLAGS.updates+update)
                        print("checkpoint saved at "+save_path)
                iter_counter+=1
            epoch_duration = time.time() - epoch_time
            print("Epoch duration: %4.4f" % (epoch_duration))
            print("estimated time to completion: %4.4f" % ((FLAGS.epoch - epoch)*(epoch_duration)/60), 'mins')

if __name__ == '__main__':
    tf.app.run()
