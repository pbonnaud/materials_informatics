#!/usr/bin/env python
# coding: utf-8
###     
### Last modification (DD/MM/YYY) : 26/07/2023
###
####################################################################################################
###                                                                                              ###
### Convolutional Neural Network (CNN) Autoencoder for Scientific Data                           ###
###                                                                                              ###
###                                                                                              ###  
### # From simple dense to complex CNN architecture: Showing the power of CNN in image           ###
###   reconstruction                                                                             ###
###                                                                                              ###
###                                                                                              ###
####################################################################################################

####################################################################################################
#                                                                                                  #
# Machine Learning for Materials Informatics                                                       #
#                                                                                                  #
# #### Markus J. Buehler, MIT                                                                      # 
#                                                                                                  #
####################################################################################################

####################################################################################################
#                                                                                                  #
# Process:  Encoder  -->  latent space (coarse-grained variables)  -->  Decoder                    #
# -------                                                                                          #
#                                                                                                  #
# ![image.png](attachment:image.png)                                                               #
#                                                                                                  #
# # Data Loading and Preprocessing                                                                 #
#                                                                                                  #
####################################################################################################

### Import libraries for the image augmentation ####################################################

import os

import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#comment this out if you would like to use the GPU

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

print (tf.test.gpu_device_name())

#sys.exit()

### Set the batch size #############################################################################

batch_size = 128

### Loads the training and test data sets (ignoring class labels) ##################################

#(x_train, _), (x_test, _) = mnist.load_data()

x_train = tf.keras.utils.image_dataset_from_directory(
          'dataset/leaf_data_augmented',
          labels=None,
          label_mode="int",
          class_names=None,
          color_mode="rgb",
          batch_size=batch_size,
          image_size=(128, 128),
          shuffle=True,
          seed=232424,
          validation_split=0.1,
          subset="training",
          interpolation="bilinear",
          follow_links=False,
          crop_to_aspect_ratio=True,
          )

x_test = tf.keras.utils.image_dataset_from_directory(
         'dataset/leaf_data_augmented',
         labels=None,
         label_mode="int",
         class_names=None,
         color_mode="rgb",
         batch_size=batch_size,
         image_size=(128, 128),
         shuffle=False,
         seed=232424,
         validation_split=0.1,
         subset="validation",
         interpolation="bilinear",
         follow_links=False,
         crop_to_aspect_ratio=True,
         )

#sys.exit()

### Plot a test image for verification that the process went well ##################################

#import matplotlib.pyplot as plt
 
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5,5))
 
for images  in x_train.take(1):

    for i in range(3):

        for j in range(3):

            ax[i][j].imshow(images[i*3+j].numpy().astype("uint8"))

            #ax[i][j].set_title(ds.class_names[labels[i*3+j]])

plt.savefig("test_D01-02-Im1.png")

#plt.show()

#sys.exit()

### This is how we iterate over all batches ########################################################

class_names = x_train.class_names                                                        # there are no classnames here!

print(class_names)

for image_batch in x_train:

    print(image_batch.shape)

#   print(image_batch)
   
print(' ');

#sys.exit() 

### Normalization of the data ######################################################################.

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = x_train.map(lambda x: (normalization_layer(x)))

normalized_ts = x_test.map(lambda x: (normalization_layer(x)))

### Set image batch and the first image ############################################################ 

image_batch = next( iter( normalized_ds ) );

first_image = image_batch[0];

# Note that pixel values are now in the range [0,1] ////////////////////////////////////////////////

print('Pixel values are now in the range [{}-{}]'.format(np.min(first_image), np.max(first_image)));

print(' ')

image_shape = first_image.shape

print('The shape of the first image is {}'.format(image_shape))

print(' ')

#sys.exit()

### Now we generate x as INPUT AND OUTPUT ##########################################################

x_train = normalized_ds.map(lambda x: (x, x))

x_test = normalized_ts.map(lambda x: (x, x))

####################################################################################################
#                                                                                                  #
# The data set consists of 3D arrays with 60K training and 10K test images. The images have a      #
# resolution of 28 x 28 (pixels) each. You can easily adapt this by loading your own dataset       #
# (we will show that tomorrow).                                                                    #
#                                                                                                  #
####################################################################################################

####################################################################################################
#                                                                                                  #
# Build a Convolutional Autoencoder                                                                #
# ---------------------------------                                                                #
#                                                                                                  #
####################################################################################################

### Set the number of epochs for the training part #################################################

itrain_epochs = 128;
#itrain_epochs = 32  # for testing

# Structure of the autoencoder: 16 * 16 * 8

autoencoder = Sequential()

# Encoder Layers ///////////////////////////////////////////////////////////////////////////////////

autoencoder.add(Conv2D(64, (2, 2), activation='relu', padding='same', input_shape=image_shape))

autoencoder.add(MaxPooling2D((2, 2), padding='same'))

autoencoder.add(Conv2D(32, (2, 2), activation='relu', padding='same'))

autoencoder.add(MaxPooling2D((2, 2), padding='same'))

autoencoder.add(Conv2D(8, (2, 2), strides=(2,2), activation='relu', padding='same'))

# Flatten encoding for visualization ///////////////////////////////////////////////////////////////

autoencoder.add(Flatten(), )
#autoencoder.add(Dense(128), )

autoencoder.summary()

###### End of the encoder oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

autoencoder.add(Reshape((16, 16, 8)))

# Decoder Layers ///////////////////////////////////////////////////////////////////////////////////

autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

autoencoder.add(UpSampling2D((2, 2)))

autoencoder.add(Conv2D(16, (2, 2), activation='relu', padding='same'))

autoencoder.add(UpSampling2D((2, 2)))

autoencoder.add(Conv2D(32, (2, 2), activation='relu', padding='same'))

autoencoder.add(UpSampling2D((2, 2)))

autoencoder.add(Conv2D(3, (4,4),  activation='sigmoid', padding='same'))

autoencoder.summary()

#sys.exit()

####################################################################################################
#                                                                                                  #
# Encoder Model:                                                                                   #
# --------------                                                                                   #
#                                                                                                  #
# Create a new `Model` with the same input as the autoencoder, but the output will be that of the  #
# flattening layer.                                                                                #
#                                                                                                  #
####################################################################################################

#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_52').output)
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten').output)

encoder.summary()

#decoder = Model(inputs=autoencoder.get_layer('reshape_48').input, outputs=autoencoder.output)
decoder = Model(inputs=autoencoder.get_layer('reshape').input, outputs=autoencoder.output)

decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, epochs=itrain_epochs, )

autoencoder.save('leaf_autoencoder')

### Load a model

#model = keras.models.load_model('leaf_autoencoder')

####################################################################################################
#                                                                                                  #
# The reconstructed digits look even better than before. This is no surprise given an even lower   #
# validation loss. Other than slight improved reconstruction, check out how the encoded image has  #
# changed. What's even cooler is that the encoded images of the 9 look similar as do those of the  #
# 8's. This similarity was far less pronounced for the simple and deep autoencoders.               #
#                                                                                                  #
####################################################################################################

encoded_imgs = encoder.predict(x_test);

decoded_imgs = autoencoder.predict(x_test);

print ('The shape of the encoded images is {}'.format(encoded_imgs.shape));

print(' ')

print ('The shape of the decoded images is {}'.format(decoded_imgs.shape));

print(' ')

#sys.exit()

### Set the number of samples for the figure #######################################################

n_samples = 8;

### Plot the figure ################################################################################

plt.figure(figsize=(n_samples*3, 4));

for i in range (n_samples):

    image_batch = next( iter(normalized_ds) );

    orig = image_batch[0];
    
    ax = plt.subplot(3, n_samples, i + 1);
    
    plt.imshow(orig);

    #plt.gray()

    ax.get_xaxis().set_visible(False);

    ax.get_yaxis().set_visible(False);
    
    
    orig = tf.expand_dims(orig, axis=0);
    
    print ('The shape of orig is {}'.format(orig.shape));
 
    print(' ');

    encoded=encoder.predict(orig)[0];                                                    # encoded_imgs[i,:]
    
    decoded = autoencoder.predict(orig)[0];                                              # decoded_imgs[i,:]

    print(encoded.shape);

    print(decoded.shape);
    
    # plot encoded image

    ax = plt.subplot(3, n_samples, n_samples + i + 1);

    #plt.imshow(encoded)

    plt.plot(encoded,'.');

    plt.gray();
     
    
    plt.gray();

    ax.get_xaxis().set_visible(False);

    ax.get_yaxis().set_visible(False);

    # plot reconstructed image

    ax = plt.subplot(3, n_samples, 2*n_samples + i + 1);

    plt.imshow(decoded);
 
    ax.get_xaxis().set_visible(False);

    ax.get_yaxis().set_visible(False);


plt.savefig("test_D01-02-Im2.png")
    
#sys.exit()

####################################################################################################
#                                                                                                  #
# Space is very high-dimensional ...                                                               #
#                                                                                                  #
####################################################################################################

### Get the PCA module from sklearn ################################################################

from sklearn.decomposition import PCA

### Get encoded image from the test set and the train set ##########################################

encoded_imgs_test = encoder.predict(x_test);

encoded_imgs_train = encoder.predict(x_train);

### Print the shape of encoded images for the test and the training sets ########################### 

print ('Shape of the test set for images {}'.format(encoded_imgs_test.shape));

print(' ');

print ('Shape of the training set for images {}'.format(encoded_imgs_train.shape));

print(' ');

#sys.exit();

### Setup PCA ######################################################################################

pca_train = PCA(n_components=2);

pca_train.fit(encoded_imgs_train);

X_pca_train = pca_train.transform(encoded_imgs_train);

print("original shape:   ", encoded_imgs_train.shape);

print(' ');

print("transformed shape:", X_pca_train.shape);

print(' ');

### Apply an inversion transformation ##############################################################

X_new = pca_train.inverse_transform(X_pca_train);

X_new.shape;

print("inverted shape:", X_new.shape)

print(' ');

### Plot results from the PCA approach #############################################################

plt.scatter(X_pca_train[:, 0], X_pca_train[:, 1], alpha=0.8,s=1,  color='blue');

#plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8, cmap='flag', s=1,  color='green')

plt.axis('equal');

plt.savefig("test_D01-02-Im3.png")

plt.show();

#sys.exit();

####################################################################################################
#                                                                                                  #
# Now we move in reduced space and explore images there ...                                        #
# ---------------------------------------------------------                                        #
#                                                                                                  #
# To do this we simply take a number of samples and invert them back into full latent space        #
#                                                                                                  #
# Alternatively, we could also generate a random set of two numbers and generate an image!         #
#                                                                                                  #
####################################################################################################

pca_train = PCA(n_components=32);

pca_train.fit(encoded_imgs_train);

X_pca_train = pca_train.transform(encoded_imgs_train);

print("original shape:   ", encoded_imgs_train.shape);

print(" ");

print("transformed shape:", X_pca_train.shape)

print(' ');

X_new = pca_train.inverse_transform(X_pca_train);

X_new.shape

print("inverted shape:", X_new.shape);

print(' ')

# Function that gets images ////////////////////////////////////////////////////////////////////////

def get_image(z,ii):
    
    #print (orig.shape) 

    #encoded=encoder.predict(orig)  [0]# encoded_imgs[i,:]
    
    z = tf.expand_dims(z, axis=0);

    print("The shape of z is {}".format(z.shape));

    print(' ')

    #z=pca_train.inverse_transform(z);

    z2 = pca_train.inverse_transform(z);

    print ('The shape of z2 is {}'.format(z2.shape));

    print(' ');

    x_hat = decoder.predict(z2)[0] # decoded_imgs[i,:]

    print ('The shape of x_hat is {}'.format(x_hat.shape));

    print(' ');

    #x_hat = autoencoder.decoder(z);

    #x_hat=torch.squeeze(x_hat).permute(1, 2, 0);

    #print (x_hat);

    plt.imshow(x_hat);

    plt.savefig("test_D01-02-Im4-{}.png".format(ii))

    plt.show();
    
    return x_hat 


### Generate images from the selected sample #######################################################

for i in range (n_samples):

    #print    (X_pca_train[i])

    #print    (X_pca_train[i].shape)
   
    get_image(X_pca_train[i],i);
   
   

print('End of the script')



