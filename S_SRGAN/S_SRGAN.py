import os, sys, pprint, time
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
import model_S_SRGAN as model
from utils import *

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("updates", int(1e4), "total number of updates [1e4]")
#flags.DEFINE_float("learning_rate", 1e-4, "initial Learning rate for adam [1e-3]")
#flags.DEFINE_float("decay_rate", .95, "learning rate decay per thousand updates - set to 1. for constant learning rate [0.95]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_integer("batch_size", 32, "The number of batch images [32]")
flags.DEFINE_integer("factor", 4, "The reduction factor [4]")
flags.DEFINE_integer("HR_size", 32, "The size of the high resolution patch to produce [32]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("save_step", 100, "The interval of saving checkpoints and generating summaries. [100]")
flags.DEFINE_string("data_dir", "../datasets/birds", "Directory name to save the image samples [../datasets/birds]")
flags.DEFINE_string("run_name", "run_", "name of the run")
flags.DEFINE_boolean("resume_training", False, "pretrain the model (true) or resume from previous checkpoint")
flags.DEFINE_integer("resume_from",0, "if resuming from previous checkpoint, resume from this update [0]")
flags.DEFINE_float("scaling", 1e3, "scaling of the content vs adversarial loss [1e3]")
FLAGS = flags.FLAGS

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    batchsize = FLAGS.batch_size
    g_scaling = FLAGS.scaling
    HR_size = FLAGS.HR_size
    LR_size = HR_size//FLAGS.factor
    c_dim = FLAGS.c_dim
    ## forcing it to use the GPU only runs into memory issues on my machine    
    # with tf.device('/GPU:0'):
    ##========================= DEFINE MODEL ===========================##
    HR_images = tf.placeholder(tf.float32, [batchsize, HR_size, HR_size, c_dim], name='HR_images')
    LR_images = tf.image.resize_images(HR_images, [LR_size, LR_size])
    sample_images =  tf.placeholder(tf.float32, [batchsize, HR_size, HR_size, c_dim], name='sample_images')
    # generator for training
    generator = model.generator(LR_images, is_train=True, reuse=False)
    generated_images = generator.outputs
    # generated fake images --> discriminator
    discriminator_fake = model.discriminator(generated_images, is_train=True, reuse=False)
    # real images --> discriminator
    discriminator_real = model.discriminator(HR_images, is_train=True, reuse=True)
    # generator for sample generation, set is_train to False so that BatchNormLayer behave differently
    samples = model.generator(tf.image.resize_images(sample_images, [LR_size, LR_size]), is_train=False, reuse=True)

    ##========================= DEFINE TRAIN OPS =======================##
    
    dr_logits = discriminator_real.outputs
    df_logits = discriminator_fake.outputs
    
    p = 1.    
    # discriminator: real images are labelled as 1
    d_loss_real = tl.cost.sigmoid_cross_entropy(dr_logits, p*tf.ones_like(dr_logits), name='dreal')
    # discriminator: images from the generator (fake) are labelled as 0
    d_loss_fake = tl.cost.sigmoid_cross_entropy(df_logits, (1-p)*tf.ones_like(df_logits), name='dfake')
    # total loss is the sum of the previous two
    d_loss = d_loss_real + d_loss_fake
    
    # adversarial generator loss: try to make the the fake images look real (1)
    g_loss_adversarial = tl.cost.sigmoid_cross_entropy(df_logits, tf.ones_like(df_logits), name='gfake')    
    g_loss_content = tf.losses.mean_squared_error(generated_images, HR_images)
    g_loss = g_loss_adversarial/g_scaling + g_loss_content
    
    # list of generator and discriminator variables for use by the optimizers
    g_vars = tl.layers.get_variables_with_name('generator', True, False)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, False)
    
    n_vars_d= len(d_vars)
    n_vars_g= len(g_vars)
    
    boundaries = [FLAGS.updates, ]
    values = [1e-5, 1e-7]

    global_step_d = tf.Variable(0, trainable=False)
    learning_rate_d = tf.train.piecewise_constant(global_step_d, boundaries, values)

    global_step_g = tf.Variable(0, trainable=False)
    learning_rate_g = tf.train.piecewise_constant(global_step_g, boundaries, values)

    d_optim = tf.train.AdamOptimizer(learning_rate_d, beta1=FLAGS.beta1).minimize(d_loss, var_list = d_vars, global_step = global_step_d)
    g_optim = tf.train.AdamOptimizer(learning_rate_g, beta1=FLAGS.beta1).minimize(g_loss, var_list = g_vars, global_step = global_step_g)
        
    with tf.name_scope('loss_summaries/'):
        tf.summary.scalar('generator_adversarial', tf.log(g_loss_adversarial))
        tf.summary.scalar('generator_perception', tf.log(g_loss_content))
        tf.summary.scalar('discriminator_total',tf.log(d_loss))
        tf.summary.image('samples', samples.outputs, max_outputs = 6)
    
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
        
        path = FLAGS.data_dir   
        shape = [HR_size, HR_size]
        
        # creates the image pipeline
        data_preproc = data_preprocessing(path, shape)

        # sets the parameters for saving the sample images
        n_display_0 = int(np.sqrt(batchsize))
        n_display_1 = batchsize//n_display_0

        ##========================= TRAIN MODELS ================================##
        
        # saves the reference images
        ref = 'training_reference'
        reference_path = run_dir+ref+'.npy'
        if not tf.gfile.Exists(reference_path):
            HR_sample = data_preproc.batch(batchsize = batchsize, single_image=False)
            tl.visualize.save_images(HR_sample, [n_display_0, n_display_1+1], run_dir+ref+'.png')
            np.save(run_dir+ref, HR_sample)
        else:
            HR_sample = np.load(reference_path)

        if FLAGS.resume_training:
            print('loading model from '+checkpoint_dir)
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
            print('model loaded')
        start_time = time.time()
        for update in range(FLAGS.resume_from, FLAGS.updates//2):
            # generate low and high-resolution image batch:
            HR_batch = data_preproc.batch(batchsize = batchsize, single_image=False)
            # train the network and generate summary statistics for tensorboard
            _ = sess.run(g_optim, feed_dict={HR_images: HR_batch, sample_images: HR_sample})
            HR_batch = data_preproc.batch(batchsize = batchsize, single_image=False)
            _ = sess.run(g_optim, feed_dict={HR_images: HR_batch, sample_images: HR_sample})
            print("update [%3d/%3d]" % (update, 2*FLAGS.updates))
            if update>0 and (np.mod(update, FLAGS.save_step) == 0):
                summary = sess.run(merged, feed_dict={HR_images: HR_batch, sample_images: HR_sample})
                writer.add_summary(summary, update)
                # save current network parameters
                saver.save(sess, checkpoint_dir, global_step = update)
                print("checkpoint saved at "+checkpoint_dir)
        saver.save(sess, pretrain_dir, global_step = update)
        print("model fully trained and saved at "+checkpoint_dir)
if __name__ == '__main__':
    tf.app.run()
