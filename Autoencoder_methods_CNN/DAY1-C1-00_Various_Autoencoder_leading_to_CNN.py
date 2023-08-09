#!/usr/bin/env python
# coding: utf-8
### 
### Last modification (DD/MM/YYY) : 11/07/2023
###
####################################################################################################
###                                                                                              ###
### This python script is an example for Autoencoder                                             ###  
###                                                                                              ###
####################################################################################################
 
####################################################################################################
#                                                                                                  #
# From simple dense to complex CNN architecture: Showing the power of CNN in image reconstruction  #
#                                                                                                  #
# The python script is adapted from the one provided in the following online course :              #
#                                                                                                  #
# Machine Learning for Materials Informatics                                                       #
#                                                                                                  #
# #### Markus J. Buehler, MIT                                                                      #
#                                                                                                  #
# Process: Encoder ---> latent space (coarse-grained variables) --> Decoder                        #
#                                                                                                  #
# Check out these additional links to learn more about [autoencoders]                              #
# (https://en.wikipedia.org/wiki/Autoencoder). This example is implemented in [Keras]              #
# (https://keras.io/). For sources, check out [the Keras blog]                                     #
# (https://blog.keras.io/building-autoencoders-in-keras.html), [this document]                     #
# (https://blog.keras.io/building-autoencoders-in-keras.html) and [Francois Chollet]               #
# (https://twitter.com/fchollet).                                                                  #
#                                                                                                  #
####################################################################################################

####################################################################################################
#                                                                                                  #
# ![image.png](attachment:image.png)                                                               #
#                                                                                                  # 
# Autoencoders perform data compression but not in the way JPEG or MPEG methods work (which make   #
# assumptions about images, sound, and video and apply compression based on the assumptions).      #
# Instead, autoencoders **learn** (automatically) a lossy compression based on the data examples   #
# fed in. Hence, the compression is specific to those examples, and they learn the unique features #
# of the data.                                                                                     #
#                                                                                                  #
# Elements of autoencoders:                                                                        #
# -------------------------                                                                        #
#                                                                                                  #
# Autoencoders require 3 things:                                                                   #
#                                                                                                  #
# 1. Encoding function - transforming data into a CG description                                   #
#                                                                                                  #
# 2. Decoding function - moving from GC description back to full representation                    #
#                                                                                                  #
# 3. Loss function: This describes the amount of information loss between the compressed and       #
#    decompressed representations of the data examples and the decompressed representation (hence  #
#    the reference to a "loss" function).                                                          #
#                                                                                                  #
# The encoding/decoding functions are typically (parametric) neural networks and are               #
# differentiable with respect to the distance function. The differentiable part enables optimizing #
# the parameters of the encoding/decoding functions to minimize the reconstruction loss.           #
#                                                                                                  #
# What Are They Good For:                                                                          #
# -----------------------                                                                          #
#                                                                                                  #
# 1. Dimension Reduction, such as in modeling/simulation of materials                              #
#                                                                                                  #
# 2. Data Visualization and Analysis                                                               #
#                                                                                                  #
# 3. Data Denoising                                                                                # 
#                                                                                                  #
# For data denoising, think PCA, but nonlinear. In fact, if the encoder/decoder functions are      #
# linear, the result spans the space of the PCA solution. The nonlinear part is useful because     #
# they can capture, for example, multimodality in the feature space, which PCA can't.              #
#                                                                                                  #
# Dimension reduction is a direct result of the lossy compression of the algorithm. It can help    #
# with denoising and **pre-training** before building another ML algorithm. It may even replace    #
# JPG or MPEG encoding... check out [this post]                                                    #
# (https://hackernoon.com/using-ai-to-super-compress-images-5a948cf09489) based on                 # 
# [a recent paper](https://arxiv.org/abs/1708.00838).                                              #
#                                                                                                  #
####################################################################################################

####################################################################################################
#                                                                                                  #
# Here we focus on three versions of an autoencoder:                                               #
# --------------------------------------------------                                               #
#                                                                                                  #
# * VARIATION 1: Simple Autoencoder                                                                #
#                                                                                                  #
# * VARIATION 2: Deep Autoencoder                                                                  #
#                                                                                                  #
# * VARIATION 3: Convolution Autoencoder                                                           #
#                                                                                                  #
#                                                                                                  #
# Data Loading and Preprocessing:                                                                  #
# -------------------------------                                                                  #
#                                                                                                  #
# We use the [MNIST data set](http://yann.lecun.com/exdb/mnist/). We start with inputing some key  #
# elements:                                                                                        #
#                                                                                                  #
####################################################################################################

### Import libraries for the computation ###########################################################

import os

import sys

# Comment the following line if you would like to use the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#from IPython.display import Image, SVG

import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import keras

from keras.datasets import mnist

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from keras import regularizers

import tensorflow as tf

### Test the availability of GPUs for the current computation ######################################

if tf.test.gpu_device_name():
 
    print('Default GPU Device:  {}'.format(tf.test.gpu_device_name()))

else:

   print("Install GPU version of TF for better performance")

print(tf.__version__)

print ( tf.test.gpu_device_name())

#sys.exit()

####################################################################################################
#                                                                                                  #
# With that out of the way, let's load the MNIST data set and scale the images to a range between  #
# 0 and 1. If you haven't already downloaded the data set, the Keras `load_data` function will     #
# download the data directly from S3 on AWS.                                                       #
#                                                                                                  #
####################################################################################################

# Loads the training and test data sets (ignoring class labels) ////////////////////////////////////

(x_train, _), (x_test, _) = mnist.load_data()

# Scales the training and test data to range between 0 and 1. //////////////////////////////////////

max_value = float(x_train.max())

x_train = x_train.astype('float32') / max_value

x_test = x_test.astype('float32') / max_value

####################################################################################################
#                                                                                                  #
# The data set consists 3D arrays with 60K training and 10K test images. The images have a         #
# resolution of 28 x 28 (pixels) each. You can easily adapt this by loading your own dataset       #
# (we will show that tomorrow).                                                                    #
#                                                                                                  #
####################################################################################################

x_train.shape, x_test.shape

####################################################################################################
#                                                                                                  #
# To work with the images as vectors, let's reshape the 3D arrays as matrices. In doing so, we'll  # 
# reshape the 28 x 28 images into vectors of length 784                                            #
#                                                                                                  #
####################################################################################################

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

(x_train.shape, x_test.shape)

# Plot a test image for verification that the process went well ////////////////////////////////////

plt.imshow(x_test[11].reshape(28, 28))

plt.gray()

plt.savefig("test1.png")

#plt.show()

#sys.exit()

# Plot another test image for verification /////////////////////////////////////////////////////////

plt.imshow(x_test[34].reshape(28, 28))

plt.gray()
 
plt.savefig("test2.png")

#plt.show()

#sys.exit()

####################################################################################################
#                                                                                                  #
# VARIATION 1: Simple Autoencoder                                                                  #
# -------------------------------                                                                  #
#                                                                                                  #
# We start with a simple autoencoder for illustration. The encoder and decoder functions are each  #
# fully-connected neural layers. The encoder function uses a [ReLU activation function]            #
# (https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), while the decoder function uses a   #
# [sigmoid activation function]                                                                    #
# (https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions).          #
#                                                                                                  #
# So what are the encoder and the decoder layers doing?                                            #
#                                                                                                  #
# * The encoder layer "encodes" the input image as a compressed representation in a reduced        #
#   dimension. The compressed image typically looks garbled, nothing like the original image.      #
#                                                                                                  #
# * The decoder layer "decodes" the encoded image back to the original dimension. The decoded      #
#   image is a [lossy reconstruction](https://en.wikipedia.org/wiki/Lossy_compression) of the      #
#   original image.                                                                                #
#                                                                                                  #
# In our example, the compressed image has a dimension of 2 (but you can change that...). The      #
# encoder model reduces the dimension from the original 784(=28x28)-dimensional vector to the      #
# encoded 2-dimensional vector. The decoder model restores the dimension from the encoded          #
# lower-dimensional representation back to the original 784-dimensional vector.                    #
#                                                                                                  #
# The compression factor is the ratio of the input dimension to the encoded dimension. In our      #
# case, the factor is `392.0 = 784 / 2`.                                                           #
#                                                                                                  #
# The `autoencoder` model maps an input image to its reconstructed image.                          #
#                                                                                                  #
####################################################################################################

# input dimension = 784 ////////////////////////////////////////////////////////////////////////////

input_dim = x_train.shape[1]

print('The input dimension is ', input_dim)

print(' ')

#sys.exit()

####################################################################################################
#                                                                                                  #
# We can choose a different number of variables for the encoded state - 32 or 2, as shown here,    #
# but you can try your own                                                                         #
#                                                                                                  #
#################################################################################################### 

# How coarse do we want to coarse-grain our model to be? ///////////////////////////////////////////

#encoding_dim = 32
encoding_dim = 2

#training epochs - increase for better results...
itrain_epochs = 100

compression_factor = float(input_dim) / encoding_dim

print("The compression factor is %s" % compression_factor)

print(' ')

#sys.exit()

autoencoder = Sequential()

autoencoder.add(
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
)

autoencoder.add(
    Dense(input_dim, activation='sigmoid')
)

autoencoder.summary()

print(' ')

#sys.exit()

####################################################################################################
#                                                                                                  #
# Encoder Model                                                                                    #
# -------------                                                                                    #
#                                                                                                  # 
# We can extract the encoder model from the first layer of the autoencoder model. The reason we    #
# want to extract the encoder model is to examine what an encoded image looks like.                #
#                                                                                                  #
####################################################################################################

input_img = Input(shape=(input_dim,))

# we extract that first layer ... //////////////////////////////////////////////////////////////////

encoder_layer = autoencoder.layers[0]

encoder = Model(input_img, encoder_layer(input_img))

encoder.summary()

#sys.exit()

####################################################################################################
#                                                                                                  #
# Now we are ready to train our first autoencoder. We'll iterate on the training data in batches   #
# over the pochs. We will also use [the Adam optimizer](https://arxiv.org/abs/1412.6980) and       #
# per-pixel binary [crossentropy](https://en.wikipedia.org/wiki/Cross_entropy) loss. The purpose   #
# of the loss function is to reconstruct an image similar to the input image.                      #
#                                                                                                  #
# NOTE: I want to call out something that may look like a typo or may not be obvious at first      #
# glance. Notice the repeat of `x_train` in `autoencoder.fit(x_train, x_train, ...)`. This implies #
# that `x_train` is both the input and output, which is exactly what we want for image             #
# reconstruction.                                                                                  #
#                                                                                                  #
# I'm running this code on a laptop, so you'll notice the training times are a bit slow (no GPU).  #
#                                                                                                  #
####################################################################################################

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=itrain_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

#sys.exit()

####################################################################################################
#                                                                                                  #
# We've successfully trained our first autoencoder. The autoencoder model can compress an MNIST    #
# digit down to a set of floating-point digits.                                                    #
#                                                                                                  #
# To check out the encoded images and the reconstructed image quality, we randomly sample 10 test  #
# images. I really like how the encoded images look. Do they make sense? No. Are they eye candy    #
# though? Most definitely.                                                                         #
#                                                                                                  #
# However, the reconstructed images are quite lossy. You can see the digits clearly, but notice    #
# the loss in image quality.                                                                       #
#                                                                                                  #
####################################################################################################

num_images = 10

np.random.seed(42)

# generate random encodings... and see what they represent /////////////////////////////////////////

random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)

decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):

    # plot original image //////////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, i + 1)

    plt.imshow(x_test[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)
    
    # plot encoded image ///////////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, num_images + i + 1)

    #plt.imshow(encoded_imgs[image_idx].reshape(8, 4))
    #plt.imshow(encoded_imgs[image_idx].reshape(2, 1))

    plt.plot (encoded_imgs[image_idx].reshape(encoding_dim, 1))

    print ("Random number pair ", i, "image ID: ", image_idx, encoded_imgs[image_idx])

    plt.gray()

    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    # plot reconstructed image /////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, 2*num_images + i + 1)

    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.savefig("test3.png")

#plt.show()

#sys.exit()

####################################################################################################
#                                                                                                  # 
# VARIATION 2: Deep Autoencoder                                                                    #
# -----------------------------                                                                    #
#                                                                                                  #
# Above, we used single fully-connected layers for both the encoding and decoding models. Instead, #
# we can stack multiple fully-connected layers to make each of the encoder and decoder functions   #
# **deep**.                                                                                        #  
#                                                                                                  #
# We will now use **3 fully-connected layers** for the encoding model with decreasing dimensions   #
# from 128 to 64 .. and encoding_dim (e.g. 32, or 2...). Likewise, we'll add 3 fully-connected     #
# decoder layers that reconstruct the image back to 784 dimensions. Except for the last layer,     #
# we'll use ReLU activation functions again.                                                       #
#                                                                                                  #
# In Keras, this is simple to do, and you can use this example for your applications. We can use   #
# the same training configuration as above (or a variation thereof - e.g. Adam + 50 epochs + batch #
# size of 256 works well).                                                                         #
#                                                                                                  #
####################################################################################################

autoencoder = Sequential()

# Encoder Layers ///////////////////////////////////////////////////////////////////////////////////

autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))

autoencoder.add(Dense(2 * encoding_dim, activation='relu'))

autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers ///////////////////////////////////////////////////////////////////////////////////

autoencoder.add(Dense(2 * encoding_dim, activation='relu'))

autoencoder.add(Dense(4 * encoding_dim, activation='relu'))

autoencoder.add(Dense(input_dim, activation='sigmoid'))

autoencoder.summary()

####################################################################################################
#                                                                                                  #
# Encoder Model                                                                                    #
# -------------                                                                                    #
#                                                                                                  #
# Like we did above, we can extract the encoder model from the autoencoder. The encoder model      #
# consists of the first 3 layers in the autoencoder, so let's extract them to visualize the        # 
# encoded images.                                                                                  #
#                                                                                                  #
####################################################################################################

# Getting the encoder ... //////////////////////////////////////////////////////////////////////////

input_img = Input(shape=(input_dim,))

encoder_layer1 = autoencoder.layers[0]

encoder_layer2 = autoencoder.layers[1]

encoder_layer3 = autoencoder.layers[2]

encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=itrain_epochs,
                batch_size=128,
                validation_data=(x_test, x_test))

####################################################################################################
#                                                                                                  #
# As with the simple autoencoder, we randomly sample 10 test images (the same ones as before). The #
# reconstructed digits look much better than those from the single-layer autoencoder. This         #
# observation aligns with the reduction in validation loss after adding multiple layers to the     #
# autoencoder.                                                                                     #
#                                                                                                  #
####################################################################################################

num_images = 10

np.random.seed(42)

random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)

decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):

    # plot original image //////////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, i + 1)

    plt.imshow(x_test[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)
    
    # plot encoded image ///////////////////////////////////////////////////////////////////////////

    #ax = plt.subplot(3, num_images, num_images + i + 1)
    #plt.plot (encoded_imgs[image_idx].reshape(2, 1))
    
    #plt.imshow(encoded_imgs[image_idx].reshape(8, 4))

    print ("Random number pair ", i, "image ID: ", image_idx, encoded_imgs[image_idx])
    
    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)


    # plot reconstructed image /////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, 2*num_images + i + 1)

    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.savefig("test3.png")

#plt.show()

#sys.exit()

####################################################################################################
#                                                                                                  # 
# VARIATION 3: Convolutional Autoencoder                                                           #
# --------------------------------------                                                           #
#                                                                                                  #
# Now that we've explored deep autoencoders, let us now use a convolutional autoencoder instead,   #
# given that the input objects are images (so now we have a deep, convolutional NN). What this     #
# means is our encoding and decoding models will be [convolutional neural networks]                #
# (http://cs231n.github.io/convolutional-networks/) instead of fully-connected networks.           #
#                                                                                                  #
# Keras makes this very easy for us. Before we get started though, we need to reshape the images   #
# back to `28 x 28 x 1` for the convolutional layers. The 1 is for 1 channel because black and     #
# white. If we had RGB color, there would be 3 channels. We now use a latent space dimension of    #
# 128.                                                                                             #
#                                                                                                  #
####################################################################################################

#itrain_epochs = 25
itrain_epochs = 35

x_train = x_train.reshape((len(x_train), 28, 28, 1))

x_test = x_test.reshape((len(x_test), 28, 28, 1))

####################################################################################################
#                                                                                                  #
# To build the convolutional autoencoder, we'll make use of `Conv2D` and `MaxPooling2D` layers for #
# the encoder and `Conv2D` and `UpSampling2D` layers for the decoder. The encoded images are       #
# transformed to a 3D array of dimensions `4 x 4 x 8`, but to visualize the encoding, we'll        #
# flatten it to a vector of length 128. I tried to use an encoding dimension of 32 like above, but #
# I kept getting subpar results.                                                                   #
#                                                                                                  #
# After the flattening layer, we reshape the image back to a `4 x 4 x 8` array before upsampling   #
# back to a `28 x 28 x 1` image.                                                                   #
#                                                                                                  #
####################################################################################################

autoencoder = Sequential()

# Encoder Layers ///////////////////////////////////////////////////////////////////////////////////

autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))

autoencoder.add(MaxPooling2D((2, 2), padding='same'))

autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

autoencoder.add(MaxPooling2D((2, 2), padding='same'))

autoencoder.add(Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same'))

# Flatten encoding for visualization ///////////////////////////////////////////////////////////////

autoencoder.add(Flatten())

###### Encoder ENDS HERE ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

autoencoder.add(Reshape((4, 4, 8)))

# Decoder Layers ///////////////////////////////////////////////////////////////////////////////////

autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

autoencoder.add(UpSampling2D((2, 2)))

autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

autoencoder.add(UpSampling2D((2, 2)))

autoencoder.add(Conv2D(16, (3, 3), activation='relu'))

autoencoder.add(UpSampling2D((2, 2)))

autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

autoencoder.summary()

####################################################################################################
#                                                                                                  #
# Encoder Model                                                                                    #
# -------------                                                                                    #
#                                                                                                  #
# To extract the encoder model for the autoencoder, we're going to use a slightly different        #
# approach than before. Rather than extracting the first 6 layers, we're going to create a new     #
# `Model` with the same input as the autoencoder, but the output will be that of the flattening    #
# layer. As a side note, this is a very useful technique for grabbing submodels for things like    #
# [transfer learning](http://ruder.io/transfer-learning/).                                         #
#                                                                                                  #
# As I mentioned before, the encoded image is a vector of length 128.                              #
#                                                                                                  #
####################################################################################################

#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten').output)

encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=itrain_epochs,
                batch_size=32,
                validation_data=(x_test, x_test))

####################################################################################################
#                                                                                                  #
# The reconstructed digits look even better than before. This is no surprise given an even lower   #
# validation loss. Other than slight improved reconstruction, check out how the encoded image has  # 
# changed. What's even cooler is that the encoded images of the 9 look similar as do those of the  #
# 8's. This similarity was far less pronounced for the simple and deep autoencoders.               #
#                                                                                                  #
####################################################################################################

num_images = 10

np.random.seed(42)

random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)

decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):

    # plot original image //////////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, i + 1)

    plt.imshow(x_test[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)
    
    # plot encoded image ///////////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, num_images + i + 1)

    plt.imshow(encoded_imgs[image_idx].reshape(16, 8))

    #print (encoded_imgs[image_idx])
        
    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    # plot reconstructed image /////////////////////////////////////////////////////////////////////

    ax = plt.subplot(3, num_images, 2*num_images + i + 1)

    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.savefig("test4.png")

plt.show()

sys.exit()

