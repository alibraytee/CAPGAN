# -*- coding: utf-8 -*-
'''
Note:
Experiment results for MNIST, Fashion_MNIST and BreakHis are generated by this CAP_GAN_Light code.

'''

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

"""### Author: Yuchong Yao
#### Contributor： Yuanbang Ma

# Create Imbalance Data
"""

random.seed(2021)

def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
        # images = tf.image.resize(images, [32,32])
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
        # images = tf.image.resize(images, [32,32])
    return images

# # Load MNIST Fashion
# from tensorflow.keras.datasets.fashion_mnist import load_data

# # Load MNIST
# from tensorflow.keras.datasets.mnist import load_data

# # Load CIFAR-10
from tensorflow.keras.datasets.cifar10 import load_data

# Load training set
(images, labels), (test_images, test_labels) = load_data()
images = change_image_shape(images)
labels = labels.reshape(-1)

# Sample classes
ratio=20

# Define the majority class
majority = 8

train_images = []
train_labels = []
for i in range(np.unique(labels).shape[0]):
    cur_images = images[labels == i]
    # divide ratio for minority
    if i != majority:
        # can shuffle for random choice
        # np.random.shuffle(cur_images)
        cur_images = cur_images[:(cur_images.shape[0])//ratio]
    train_images += list(cur_images)
    train_labels += [i]*cur_images.shape[0]
images = np.array(train_images)
labels = np.array(train_labels)
shuffler = np.random.permutation(len(images))

train_images = images[shuffler]
train_labels = labels[shuffler]
test_images = change_image_shape(test_images)
test_labels = test_labels.reshape(-1)

print(train_images.shape)
print(train_labels.shape)
print(np.unique(labels))

####--------------------------####
####  For BreakHis and Cells  ####
####     Scale in [0,255]     ####
####--------------------------####

# Train test split, for autoencoder (actually, this step is redundant if we already have test set)
#train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, shuffle=True, random_state=42)

"""# Prameters"""

latent_dim=256
trainRatio=5
n_classes=10
discriminator_extra_steps = 3
# epochs and learning steps
cvae_epoch = 30
gan_learning_steps = 20

"""# Random Over Sampling"""

majority_size = len(np.where(labels == majority)[0])
resample_train_images = []
resample_train_labels = []
for c in np.unique(labels):
    c_images = images[labels == c]
    c_labels = labels[labels == c] 
    if c != majority:  
        ids = np.arange(len(c_images))
        choices = np.random.choice(ids, majority_size)
        c_images = c_images[choices,]
        c_labels = c_labels[choices,]
    resample_train_images += list(c_images)
    resample_train_labels += list(c_labels)
        

resample_train_images = np.array(resample_train_images)
resample_train_labels = np.array(resample_train_labels)
shuffler = np.random.permutation(len(resample_train_images))
train_images = resample_train_images[shuffler]
train_labels = resample_train_labels[shuffler]  
print(train_images.shape, train_labels.shape)

"""# Preprocessing"""

# Set channel
channel = images.shape[-1]

# to 64 x 64 x channel
real = np.ndarray(shape=(train_images.shape[0], 32, 32, channel))
for i in range(train_images.shape[0]):
    real[i] = cv2.resize(train_images[i], (32, 32)).reshape((32, 32, channel))

test_real = np.ndarray(shape=(test_images.shape[0], 32, 32, channel))
for i in range(test_images.shape[0]):
    test_real[i] = cv2.resize(test_images[i], (32, 32)).reshape((32, 32, channel))

train_images = real.astype('float32')/255.
test_images = test_real.astype('float32')/255.

input_shape = (32,32,3)

print(train_images.shape)
print(test_images.shape)

"""# Models Setup"""

#train_size = 6000
#test_size = 1000
batch_size = 128
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


    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation=LeakyReLU(), name="encode_conv1")(inputs)
    # x = tf.keras.layers.BatchNormalization()(x) ## ?
    # x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(), name="encode_conv2")(x)
    # x = tf.keras.layers.BatchNormalization()(x) ## ?
    # x = tf.keras.layers.LayerNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="encode_conv3")(x)
    # x = tf.keras.layers.BatchNormalization()(x) ## ?
    # x = tf.keras.layers.LayerNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="encode_conv4")(x)
    # x = tf.keras.layers.BatchNormalization()(x) ## ?
    # x = tf.keras.layers.LayerNormalization()(x)
    conv_shape = x.shape
    # print(conv_shape)


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
    # print(units)
    x = tf.keras.layers.Dense(units, activation = LeakyReLU(), name="decode_dense1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    # x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")(x)
    # print(x.shape)

    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="decode_conv2d_1")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    # print(x.shape)  

    # x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="decode_conv2d_2")(x)
    x = tf.keras.layers.BatchNormalization()(x) ## ?
    # print(x.shape)
    # x = tf.keras.layers.LayerNormalization()(x)
    # x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(), name="decode_conv2d_3")(x)
    # x = tf.keras.layers.BatchNormalization()(x) ## ?
    # x = tf.keras.layers.LayerNormalization()(x)
    outputs = tf.keras.layers.Conv2DTranspose(filters=n_channel, kernel_size=3, strides=1, padding='same', activation='sigmoid', name="decode_final")(x)
    # print(outputs.shape)

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

unique_labels = []
for i in range(n_classes):
    unique_labels.append(np.where(test_labels==i)[0][0])
test_sample = test_images[unique_labels]
test_label_sample = test_labels[unique_labels]
#test_sample = test_images[:10]
#test_label_sample = test_labels[:10]
def generate_and_save_images(model, epoch, test_sample, test_label_sample):
    decoded_imgs = model([test_sample, test_label_sample])
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        
    # display original
        ax = plt.subplot(2, n, i + 1)
        # plt.imshow(test_sample[i].reshape(32,32))
        plt.imshow(test_sample[i])
        # plt.gray()
        # plt.imshow(test_sample[i]*0.5+0.5)
        plt.title("original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].numpy().reshape(32,32))
        plt.imshow(decoded_imgs[i]) # [0, 1]
        # plt.gray()
        # plt.imshow(decoded_imgs[i]*0.5 + 0.5) # [-1, 1]
        plt.title("reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9)
loss_metric = tf.keras.metrics.Mean()
ce_loss = tf.keras.losses.BinaryCrossentropy()
mse_loss = tf.keras.losses.MeanSquaredError()
encoder, embedder, decoder, vae = get_models(input_shape=input_shape, latent_dim=latent_dim, n_classes=n_classes)

def compute_loss(model, x, y):
    
    reconstructed = model([x, y])
    kl = sum(model.losses)
    mse = mse_loss(tf.reshape(x, shape=[-1]), tf.reshape(reconstructed, shape=[-1])) 
    ce = ce_loss(tf.reshape(x, shape=[-1]), tf.reshape(reconstructed, shape=[-1])) * x.shape[0] * x.shape[1]
    # ce = tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(x, shape=[-1]), tf.reshape(reconstructed, shape=[-1])) * x.shape[0] * x.shape[1]
    return  kl + ce + mse

@tf.function
def train_step(model, x, y, optimizer):

    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
epochs = 30
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

# Refer to the WGAN-GP Architecture. https://github.com/keras-team/keras-io/blob/master/examples/generative/wgan_gp.py
class CAP_GAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(CAP_GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.train_ratio = trainRatio
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(CAP_GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        if isinstance(data, tuple):
            real_images = data[0]
            labels = data[1]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        ########################### Train the Discriminator ###########################
        # For each batch, we are going to perform cwgan-like process
        for i in range(self.train_ratio):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
            wrong_labels = tf.random.uniform((batch_size,), 0, n_classes)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, fake_labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, fake_labels], training=True)
                # Get the logits for real images
                real_logits = self.discriminator([real_images, labels], training=True)
                # Get the logits for wrong label classification
                wrong_label_logits = self.discriminator([real_images, wrong_labels], training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits,
                                        wrong_label_logits=wrong_label_logits
                                        )

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        ########################### Train the Generator ###########################
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, fake_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, fake_labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

# Optimizer for both the networks
# learning_rate=0.0002, beta_1=0.5, beta_2=0.9 are recommended
generator_optimizer = Adam(
    learning_rate=0.0005, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = Adam(
    learning_rate=0.0005, beta_1=0.5, beta_2=0.9
)


# We refer to the DRAGAN loss function. https://github.com/kodalinaveen3/DRAGAN
# Define the loss functions to be used for discrimiator
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_logits, fake_logits, wrong_label_logits):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    wrong_label_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_label_logits, labels=tf.zeros_like(fake_logits)))

    return wrong_label_loss + fake_loss + real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_logits):
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return fake_loss

def generator_label(embedder, decoder, latent_dim):
    

    label = Input((1,), dtype='int32')
    latent = Input((latent_dim,))

    labeled_latent = embedder([latent, label])
    gen_img = decoder(labeled_latent)
    model = Model([latent, label], gen_img)

    return model


def build_discriminator(encoder, img_size, n_classes):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    inter_output_model = Model(inputs=encoder.input, outputs=encoder.layers[-3].output)
    x = inter_output_model(img)

    le = Flatten()(tf.keras.layers.Embedding(n_classes, 512)(label))
    le = Dense(latent_dim)(le)
    le = LeakyReLU(0.2)(le)
    x_y = multiply([x, le])
    x_y = Dense(512)(x_y)

    out = Dense(1)(x_y)
    model = Model(inputs=[img, label], outputs=out)

    return model

def discriminator_cwgan():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    img = Input(input_shape)
    label = Input((1,), dtype='int32')


    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(img)
    # x = LayerNormalization()(x) # It is not suggested to use BN in Discriminator of WGAN
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Flatten()(x)
    # print(x.shape)

    le = Flatten()(Embedding(n_classes, 512)(label))
    le = Dense(1024)(le)
    # print(le.shape)
    le = LeakyReLU(0.2)(le)
    x_y = multiply([x, le])
    x_y = Dense(512)(x_y)

    out = Dense(1)(x_y)

    model = Model(inputs=[img, label], outputs=out)

    return model

"""# Compile Models"""

d_model = build_discriminator(encoder, img_size=input_shape, n_classes=n_classes)  
# d_model = discriminator_cwgan()  # without initialization
g_model = generator_label(embedder, decoder, latent_dim=latent_dim)  

trainRatio = 3

cap_vae = CAP_GAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
    discriminator_extra_steps=discriminator_extra_steps,
)

# Compile the model
cap_vae.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

cap_vae.train_ratio

"""# Plot During Training"""

# Plot/save generated images through training
def plt_img(generator, epoch):
    np.random.seed(42)
    latent_gen = np.random.normal(size=(n_classes, latent_dim))

    n = n_classes

    plt.figure(figsize=(2*n, 2*(n+1)))
    for i in range(n):
        # display original
        ax = plt.subplot(n+1, n, i + 1)
        if test_images[np.where(test_labels==i)[0]][4].shape[-1] ==3:
            plt.imshow(test_images[np.where(test_labels==i)[0]][4])
            # plt.gray()
            
        else:
            plt.imshow(test_images[np.where(test_labels==i)[0]][4].reshape(32, 32))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for c in range(n):
            decoded_imgs = generator.predict([latent_gen, np.ones(n)*c])
            # decoded_imgs = decoded_imgs * 0.5 + 0.5
            # display generation
            ax = plt.subplot(n+1, n, (i+1)*n + 1 + c)
            if decoded_imgs[i].shape[-1]==3:
                plt.imshow(decoded_imgs[i])
                # plt.gray()
            else:
                plt.imshow(decoded_imgs[i].reshape(32, 32))
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('cap_vae_results/generated_plot_%d.png' % epoch)
    plt.show()
    return

# make directory to store results
os.system('mkdir -p cap_vae_results')

# Record the loss
d_loss_history = []
g_loss_history = []

"""# Start Training"""

LEARNING_STEPS = 30
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step + 1, '-' * 50)
    start = time.time()
    cap_vae.fit(train_images, train_labels, batch_size=128, epochs=2)
    d_loss_history += cap_vae.history.history['d_loss']
    g_loss_history += cap_vae.history.history['g_loss']
    print("Time for this step", time.time()-start)
    if (learning_step+1)%1 == 0:
        # display.clear_output(wait=False)
        print("Time for this step", time.time()-start)
        plt_img(cap_vae.generator, learning_step)

"""# Display performance"""

# plot loss of G and D
plt.plot(d_loss_history, label='D')
plt.plot(g_loss_history, label='G')
plt.legend()
plt.show()

# save gif
import imageio
ims = []
for i in range(LEARNING_STEPS):
    fname = 'generated_plot_%d.png' % i
    dir = 'cap_vae_results/'
    if fname in os.listdir(dir):
        print('loading png...', i)
        im = imageio.imread(dir + fname, 'png')
        ims.append(im)
print('saving as gif...')
imageio.mimsave(dir + 'training_demo.gif', ims, fps=3)

"""# FID & SSIM"""

import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
import tensorflow as tf
from tensorflow.keras import models
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import gc

def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images

# generate images
def generate_images(n_classes, latent_dim, generator, steps, output_dim, **kwargs):
    
    # np.random.seed(42)
    images = [[] for _ in range(n_classes)]
    
    for _ in range(steps):
        for c in range(n_classes):
            latent_gen = np.random.normal(size=(n_classes,latent_dim))
            decoded_imags = generator.predict([latent_gen, np.ones(n_classes)*c])
            # decoded_imags = decoded_imags *0.5+0.5
            decoded_imags = tf.image.resize(decoded_imags,output_dim).numpy()
            
            loc = c%n_classes
            images[loc] +=list(decoded_imags)
    
    np_images = []
    for i in range(n_classes):
        np_images.append(np.array(images[i]))
    
    return np_images

# SSIM
def calculate_ssim(n_classes, real_images, real_labels, gen_images,steps = 10):
    
    scores = []
    # test for each class
    for n in range(n_classes):
        real_image = real_images[real_labels== n]
        gen_image = gen_images[n]
        
        real_idx = np.random.choice(real_image.shape[0], gen_image.shape[0])
        real_image = real_image[real_idx]
        
        score = 0
        for _ in range(steps):
            
            ssim = tf.image.ssim(real_image,gen_image,max_val = 1.0)
            score += np.average(ssim.numpy())
            
            shuffle(real_image)
            shuffle(gen_image)
            
        score /= steps
        print('>>> SSIM(%d): %.5f'%(n, score))
        scores += [scores]
    
    return scores

# FID
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)*255
        # store
        images_list.append(new_image)
    return asarray(images_list)

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(224, 224, 3))

def fid(n_classes, real_images, real_labels, gen_images, model, **kwargs):
    
    fid_scores = []
    
    for n in range(n_classes):
        real_image = real_images[real_labels== n]
        gen_image = gen_images[n]
        
        # slice same size with real images
        gen_image = gen_image[:real_image.shape[0]]
        
        # # Only for Dataset "Cell"
        #real_idx = np.random.choice(real_image.shape[0], gen_image.shape[0])
        #real_image = real_image[real_idx]
        
        real_image = scale_images(real_image,(224,224,3))
        gen_image = scale_images(gen_image,(224,224,3))
        
        
        real_image = preprocess_input(real_image)
        gen_image = preprocess_input(gen_image)
        
        score = calculate_fid(model, gen_image, real_image)
        print('>>> FID(%d): %.5f'%(n, score))
        fid_scores += [score]
        gc.collect()
    
    return fid_scores

###------------------------------------ ##
###             Set parameter           ##
###  Remember to change number of class ##
###------------------------------------ ##
#n_classes = n_classes
#latent_dim= latent_dim
generator = cap_vae.generator

###-----------------------------##
###       Set Real Images       ##
###  Remember to choose dataset ##
###-----------------------------##

# # Load MNIST Fashion
#from tensorflow.keras.datasets.fashion_mnist import load_data

# # Load MNIST
#from tensorflow.keras.datasets.mnist import load_data

# # Load CIFAR-10
from tensorflow.keras.datasets.cifar10 import load_data

# # Load test set
# # We use the test set as input
(_, _), (real_images, real_labels) = load_data()
real_images = change_image_shape(real_images).astype('float32') / 255
real_labels = real_labels.reshape(-1)


####--------------------------------####
#### Only for the dataset "Cells"   ####
####--------------------------------####

#real_images = np.load('./cells_train/cells_images.npy')
#real_labels = np.load('./cells_train/cells_labels.npy')

####-----------------------------------####
#### Only for the dataset "BreakHis"   ####
#### make sure the scale is in [0,1]   ####
####-----------------------------------####

#real_images = ...
#real_labels = ...

###---------------------------------##
###      Get Generated images       ##
###    Remember to adjust steps     ##
###  Remember to adjust "output_dim"##
###---------------------------------##
output_dim = [32,32]
gen_images = generate_images(n_classes=n_classes,latent_dim=latent_dim, generator=generator, steps=100, output_dim=output_dim)

plt.imshow(gen_images[9][68])

###-------##
###  FID  ##
###-------##
fid_socres = fid(n_classes,real_images,real_labels,gen_images,model)

###--------##
###  SSIM  ##
###--------##
ssim_scores = calculate_ssim(n_classes,real_images,real_labels,gen_images,steps=10)