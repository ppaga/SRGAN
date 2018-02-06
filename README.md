# SRGAN
This is an implementation of the SRGAN (https://arxiv.org/abs/1609.04802) paper in tensorflow (using tensorlayer to build the network). 

## What this project is and how it came about
This project is mainly for the purpose of getting some tensorflow practice. Coming from keras and sicne I didn't know much about tensorlayer (or tensorflow for that matter) I used the code on https://github.com/tensorlayer/dcgan/  as a template and modified it into an SRGAN. Additionally, I added things like loading variables from checkpoints, tensorboard functionality, etc... (generally trying to use as much native tensorflow as possible)

## The network and how to train it
The network itself is mostly faithful to the original, but the number of channels and the batchsize has on occasions been modified to accomodate the limited memory of my poor GTX 1050. I'm still trying to optimize the training so that it takes a reasonable amount of time, but since training the network for weeks on end isn't an option the quality will always be suboptimal. I've been training it on the Caltech-UCSD Birds 200 dataset (http://www.vision.caltech.edu/visipedia/CUB-200.html). I haven't tried the resulting network on images of other things. The training is done by calling 
python main.py
on the command line, and there are a number of optional arguments described in main.py using tensorflow flags (e.g. initial learning rate etc). Once the training starts, the checkpoints will be in ./checkpoints, the tensorboard logs in ./logs, and some sample images will be generated in ./samples. The samples come with their reference low-resolution images (the image which is passed to the network) as training_LR, and their high-resolution reference (the image which the network is trying to reconstruct).

## What is yet to be done
- There is a limited functionality to test out the network on user-provided images in the file SR.py, but basically it still needs to be implemented.
- At current I'm using a fixed and fairly steep decay which halves the learning rate every 1e3 updates. I need to implement a more flexible way of doing things (i.e. I need to create another flag for the decay rate)
- I need to post results.
