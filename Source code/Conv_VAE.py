""" 


Author: Yuchong Yao


"""

import os
import random
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, \
    Concatenate, multiply, Flatten, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, Activation, LeakyReLU, LayerNormalization, BatchNormalization

"""Models Setup"""

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
def preprocess_images(images):
  images = images.reshape((images.shape[0], 32, 32, 3)) / 255.
  return images.astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_size = 60000
batch_size = 32
test_size = 10000
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)
train_label = tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
test_label = tf.data.Dataset.from_tensor_slices(test_labels).batch(batch_size)

class Reparameterize(tf.keras.layers.Layer):
    def call(self, inputs):
  
        mu, sigma = inputs

        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * sigma) * epsilon

def Encoder(input_shape, latent_dim):

 
    inputs = tf.keras.layers.Input(shape=input_shape)


    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same", activation=LeakyReLU(), name="encode_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="encode_conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="encode_conv3")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="encode_conv4")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    conv_shape = x.shape


    x = tf.keras.layers.Flatten(name="encode_flatten")(x)
    mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = tf.keras.layers.Dense(latent_dim, name ='latent_sigma')(x)


    z = Reparameterize()((mu, sigma))

    model = tf.keras.Model(inputs, outputs=[mu, sigma, z])
    return model, conv_shape

def Embedder(latent_dim, n_classes):

    label = tf.keras.layers.Input((1,), dtype='int32')
    noise = tf.keras.layers.Input(shape=(latent_dim,))
    # ne = Dense(256)(noise)
    # ne = LeakyReLU(0.2)(ne)

  

    embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(n_classes, latent_dim)(label))
    # le = Dense(256)(le)
    # le = LeakyReLU(0.2)(le)

    embedding = tf.keras.layers.multiply([noise, embedding])
    # noise_le = Dense(latent_dim)(noise_le)

    model = tf.keras.Model(inputs=[noise, label], outputs=embedding)

    return model

def Decoder(latent_dim, conv_shape, n_channel):

    inputs = tf.keras.layers.Input(shape=(latent_dim,))

    units = conv_shape[1] * conv_shape[2] * conv_shape[3]
    x = tf.keras.layers.Dense(units, activation = LeakyReLU(), name="decode_dense1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x) ## ?

    x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")(x)


    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="decode_conv2d_1")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="decode_conv2d_2")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="decode_conv2d_3")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    outputs = tf.keras.layers.Conv2DTranspose(filters=n_channel, kernel_size=4, strides=2, padding='same', activation='sigmoid', name="decode_final")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def kl_reconstruction_loss(inputs, outputs, mu, sigma):

    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss) * -0.5
    return kl_loss

def VAE(encoder, embedder, decoder, input_shape):
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    labels = tf.keras.layers.Input((1,), dtype='int32')

    mu, sigma, z = encoder(inputs)
    labeled_z = embedder([z, labels])
    reconstructed = decoder(labeled_z)
    model = tf.keras.Model(inputs=[inputs, labels], outputs=reconstructed)
    kl = kl_reconstruction_loss(inputs, labeled_z, mu, sigma)
    model.add_loss(kl)
    return model

def get_models(input_shape, latent_dim, n_classes):

    encoder, conv_shape = Encoder(latent_dim=latent_dim, input_shape=input_shape)
    embedder = Embedder(latent_dim=latent_dim, n_classes=n_classes)
    decoder = Decoder(latent_dim=latent_dim, conv_shape=conv_shape, n_channel=input_shape[2])
    vae = VAE(encoder, embedder, decoder, input_shape=input_shape)
    return encoder, embedder, decoder, vae

test_sample = test_images[:10]
test_label_sample = test_labels[:10]
def generate_and_save_images(model, epoch, test_sample, test_label_sample):
    decoded_imgs = model([test_sample, test_label_sample])
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        
    # display original
        ax = plt.subplot(2, n, i + 1)
    #   plt.imshow(test_images[i].reshape(28,28))
        plt.imshow(test_sample[i])
        plt.title("original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
#   plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9)
loss_metric = tf.keras.metrics.Mean()
ce_loss = tf.keras.losses.BinaryCrossentropy()
mse_loss = tf.keras.losses.MeanSquaredError()
encoder, embedder, decoder, vae = get_models(input_shape=(32,32,3), latent_dim=64, n_classes=10)

def compute_loss(model, x, y):
    
    reconstructed = model([x, y])
    kl = sum(model.losses)
    mse = mse_loss(tf.reshape(x, shape=[-1]), tf.reshape(reconstructed, shape=[-1])) * x.shape[0] * x.shape[1]
    ce = ce_loss(tf.reshape(x, shape=[-1]), tf.reshape(reconstructed, shape=[-1])) * x.shape[0] * x.shape[1]
    return  kl + ce + mse

@tf.function
def train_step(model, x, y, optimizer):

    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
epochs = 100
generate_and_save_images(vae, 0, test_sample, test_label_sample)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x, train_y in zip(train_dataset, train_label):
        train_step(vae, train_x, train_y, optimizer)
        end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x, test_y in zip(test_dataset, test_label):
        loss(compute_loss(vae, test_x, test_y))
    l = loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}'.format(epoch, l, end_time - start_time))
    generate_and_save_images(vae, epoch, test_sample, test_label_sample)
