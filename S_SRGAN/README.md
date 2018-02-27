# S_SRGAN
I was inspired by my memory and running time issues to make a "smaller" version of SRGAN, where the upscaled patches would be of size 8x8 instead of 32x32. This allows for much faster training and larger batch sizes. As it turns out, it also gives much less distorted results.

The main drawback of using small patches is the limited contextual information the model has to work with (indeed, the resulting upscaled images are so small I can't even process them through an imagenet-pretraind network to compute a perception loss, I could try CIFAR but I doubt it would be very useful), which limits the quality of possible results as well as the severity of possible mistakes. So it's kind of expected that the result would be similar to something like bicubic upsampling, with some of the usual artefacts.


