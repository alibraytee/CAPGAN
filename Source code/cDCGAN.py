# -*- coding: utf-8 -*- 


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

"""Fix Seeds"""

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)

"""Data Preparation"""

def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images

"""Fashion MNIST"""

# # Load MNIST Fashion
from tensorflow.keras.datasets.fashion_mnist import load_data
# # Load CIFAR-10
# from tensorflow.keras.datasets.cifar10 import load_data

# # Load training set
(images, labels), (_,_) = load_data()
images = change_image_shape(images)
labels = labels.reshape(-1)

# Sample classes
random.seed(2021)
ratio=1

train_images = []
train_labels = []
for i in range(np.unique(labels).shape[0]):
    cur_images = images[labels == i]
    # divide ratio for minority
    if i != 8:
        # can shuffle for random choice
        # np.random.shuffle(cur_images)
        cur_images = cur_images[:(cur_images.shape[0])//ratio]
    train_images += list(cur_images)
    train_labels += [i]*cur_images.shape[0]
images = np.array(train_images)
labels = np.array(train_labels)

print(images.shape)
print(labels.shape)
print(np.unique(labels))

"""Preprocessing"""

# Set channel
channel = images.shape[-1]

# to 64 x 64 x channel
real = np.ndarray(shape=(images.shape[0], 32, 32, channel))
for i in range(images.shape[0]):
    real[i] = cv2.resize(images[i], (32, 32)).reshape((32, 32, channel))

# Train test split, for autoencoder (actually, this step is redundant if we already have test set)
x_train, x_test, y_train, y_test = train_test_split(real, labels, test_size=0.3, shuffle=True, random_state=42)

# It is suggested to use [-1, 1] input for GAN training
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

# Get image size
img_size = x_train[0].shape
# Get number of classes
n_classes = len(np.unique(y_train))

"""Hyperparameters"""

optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
latent_dim=64
# trainRatio === times(Train D) / times(Train G)
trainRatio = 1

"""Models Setup"""

def G():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    noise_le = Input((latent_dim,))

    x = Dense(2*2*256)(noise_le)
    x = LeakyReLU(alpha=0.2)(x)

    ## Size: 4 x 4 x 256
    x = Reshape((2, 2, 256))(x)

    ## Size: 8 x 8 x 128
    x = Conv2DTranspose(filters=128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 16 x 16 x 128
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 32 x 32 x 64
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 64 x 64 x 3
    generated = Conv2DTranspose(channel, (4, 4), strides=(2, 2), padding='same', activation='tanh', kernel_initializer=init)(x)


    generator = Model(inputs=noise_le, outputs=generated)
    return generator


def D():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    img = Input(img_size)

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
    # print(x.shape)

    # 4 x 4 x 256
    feature = Flatten()(x)

    feature = Dense(latent_dim)(feature)
    out = LeakyReLU(0.2)(feature)

    model = Model(inputs=img, outputs=out)
    return model

# Build Embedding model
def embedding_labeled_latent():
    # # weight initialization
    # init = RandomNormal(stddev=0.02)

    label = Input((1,), dtype='int32')
    noise = Input((latent_dim,))
    # ne = Dense(256)(noise)
    # ne = LeakyReLU(0.2)(ne)

    le = Flatten()(Embedding(n_classes, latent_dim)(label))
    # le = Dense(256)(le)
    # le = LeakyReLU(0.2)(le)

    noise_le = multiply([noise, le])
    # noise_le = Dense(latent_dim)(noise_le)

    model = Model([noise, label], noise_le)

    return model




dis = D()
gen = G()
em = embedding_labeled_latent()

# Refer to the WGAN-GP Architecture. https://github.com/keras-team/keras-io/blob/master/examples/generative/wgan_gp.py

class cDCGAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(cDCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.train_ratio = trainRatio
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(cDCGAN, self).compile()
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
        for i in range(1):
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
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
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

# build generator with pretrained decoder and embedding
def generator_label(embedding, decoder):
    # # Embedding model needs to be trained along with GAN training
    # embedding.trainable = False

    label = Input((1,), dtype='int32')
    latent = Input((latent_dim,))

    labeled_latent = embedding([latent, label])
    gen_img = decoder(labeled_latent)
    model = Model([latent, label], gen_img)

    return model

# Build discriminator with pre-trained Encoder
def build_discriminator(encoder):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    inter_output_model = Model(inputs=encoder.input, outputs=encoder.layers[-3].output)
    x = inter_output_model(img)

    le = Flatten()(Embedding(n_classes, 512)(label))
    le = Dense(2 * 2 * 256)(le)
    le = LeakyReLU(0.2)(le)
    x_y = multiply([x, le])
    x_y = Dense(512)(x_y)

    out = Dense(1)(x_y)

    model = Model(inputs=[img, label], outputs=out)

    return model

"""Compile Models"""

d_model = build_discriminator(dis)  
g_model = generator_label(em, gen)  

cdcgan = cDCGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
    discriminator_extra_steps=3,
)

# Compile the model
cdcgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)
cdcgan.train_ratio

"""plot during training"""

# Plot/save generated images through training
def plt_img(generator, epoch):
    np.random.seed(42)
    latent_gen = np.random.normal(size=(n_classes, latent_dim))

    x_real = x_test * 0.5 + 0.5
    n = n_classes

    plt.figure(figsize=(2*n, 2*(n+1)))
    for i in range(n):
        # display original
        ax = plt.subplot(n+1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test==i][4].reshape(32, 32, channel))
        else:
            plt.imshow(x_real[y_test == i][4].reshape(32, 32))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for c in range(n):
            decoded_imgs = generator.predict([latent_gen, np.ones(n)*c])
            decoded_imgs = decoded_imgs * 0.5 + 0.5
            # display generation
            ax = plt.subplot(n+1, n, (i+1)*n + 1 + c)
            if channel == 3:
                plt.imshow(decoded_imgs[i].reshape(32, 32, channel))
            else:
                plt.imshow(decoded_imgs[i].reshape(32, 32))
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('cdcgan_results/generated_plot_%d.png' % epoch)
    plt.show()
    return

# make directory to store results
os.system('mkdir -p cdcgan_results')

# Record the loss
d_loss_history = []
g_loss_history = []

"""Start training"""

LEARNING_STEPS = 20
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step + 1, '-' * 2)
    cdcgan.fit(x_train, y_train, batch_size=128, epochs=2)
    d_loss_history += cdcgan.history.history['d_loss']
    g_loss_history += cdcgan.history.history['g_loss']
    if (learning_step+1)%1 == 0:
        plt_img(cdcgan.generator, learning_step)

"""Display performance"""

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
    dir = 'cdcgan_results/'
    if fname in os.listdir(dir):
        print('loading png...', i)
        im = imageio.imread(dir + fname, 'png')
        ims.append(im)
print('saving as gif...')
imageio.mimsave(dir + 'training_demo.gif', ims, fps=3)

"""### SSIM & FID"""

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

def generate_images(n_classes, latent_dim, generator, steps, output_dim, **kwargs):
    
    # np.random.seed(42)
    images = [[] for _ in range(n_classes)]
    
    for _ in range(steps):
        for c in range(n_classes):
            latent_gen = np.random.normal(size=(n_classes,latent_dim))
            decoded_imags = generator.predict([latent_gen, np.ones(n_classes)*c])
            decoded_imags = decoded_imags *0.5+0.5
            decoded_imags = tf.image.resize(decoded_imags,output_dim).numpy()
            
            loc = c%n_classes
            images[loc] +=list(decoded_imags)
    
    np_images = []
    for i in range(n_classes):
        np_images.append(np.array(images[i]))
    
    return np_images

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
    
    return fid_scores

###------------------------------------ ##
###             Set parameter           ##
###  Remember to change number of class ##
###------------------------------------ ##
n_classes = 10
latent_dim= 64
generator = cdcgan.generator

###-----------------------------##
###       Set Real Images       ##
###  Remember to choose dataset ##
###-----------------------------##

# # Load MNIST Fashion
from tensorflow.keras.datasets.fashion_mnist import load_data

# # Load MNIST
#from tensorflow.keras.datasets.mnist import load_data

# # Load CIFAR-10
#from tensorflow.keras.datasets.cifar10 import load_data

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

###----------------------------##
###    Get Generated images    ##
###  Remember to adjust steps  ##
###----------------------------##
gen_images = generate_images(n_classes=n_classes,latent_dim=latent_dim, generator=generator, steps=100, output_dim=[28,28])

###--------##
###  SSIM  ##
###--------##
ssim_scores = calculate_ssim(n_classes,real_images,real_labels,gen_images,steps=10)

###-------##
###  FID  ##
###-------##
fid_socres = fid(n_classes,real_images,real_labels,gen_images,model)
