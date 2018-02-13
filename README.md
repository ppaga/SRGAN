# SRGAN
This is an implementation of the SRGAN (https://arxiv.org/abs/1609.04802) paper in tensorflow (using tensorlayer to build the network). 

## What this project is and how it came about
This project is mainly for the purpose of getting some tensorflow practice. Coming from keras and sicne I didn't know much about tensorlayer (or tensorflow for that matter) I used the code on https://github.com/tensorlayer/dcgan/ as a template and modified it into an SRGAN. Additionally, I added things like loading variables from checkpoints, tensorboard functionality, etc... (generally trying to use as much native tensorflow as possible). I also took code from https://github.com/antlerros/tensorflow-fast-neuralstyle/ for the initial formatting of the tensor for use by vgg.

## The network and how to train it
The network itself is mostly faithful to the original, but the number of channels and the batchsize has on occasions been modified to accomodate the limited memory of my poor GTX 1050. I follow the original paper by pre-training the network on MSE for 10^5 updates with a learning rate of 10^-4, followed by 10^5 updates with a learning rate of 10^-5, then training the network with the GAN and perception losses with the same schedule. I've been training it on the Caltech-UCSD Birds 200 dataset (http://www.vision.caltech.edu/visipedia/CUB-200.html). I haven't tried the resulting network on images of other things. The training is done by calling 

python main.py

on the command line, and there are a number of optional arguments which are shown in main.py using tensorflow flags (e.g. initial learning rate etc). Once the training starts, the checkpoints will be in ./runs/<run_name>/checkpoints, the tensorboard logs in ./logs, and some sample images will be generated in ./runs/<run_name>/samples. The pretraining has checkpoints in ./pretrain, and if the model is called without pretraining it will be necessary to move the pretrained network to the checkpoints directory. 
During training, the program will save reference high-resolution images as numpy arrays in train.npy and a visualization in train.png. If training is interrupted and resumed, it will load the npy file. If the file doesn't exist, it will generate a new batch and save it.

## How to use the trained network
The file SR.py is used to generate upscaled version of images of arbitrary size. By default, it assumes the image to be upscaled is in the same directory as SR.py and is called 'image', but you can pass a different path as an option. It requires you to pass the checkpoint directory of the model you want to use as an argument, i.e. it needs to be called as
python SR.py --checkpoint_dir='your path'
since it needs to load the model from there.

## What is yet to be done
- finish training
- test
- post pretty pictures
- decide on whether to include a tanh layer as the final activation function of the generator
