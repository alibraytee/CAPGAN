 
# CycleGAN

Author: Yuchong Yao
"""

# reference https://github.com/Katexiang/CycleGAN, https://www.tensorflow.org/tutorials/generative/cyclegan

"""## Set up the input pipeline"""

!pip install git+https://github.com/tensorflow/examples.git

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

AUTOTUNE = tf.data.AUTOTUNE

!unzip "/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Blood Cell Images.zip" -d "/content"

dict_characters = {0:'NEUTROPHIL',1:'EOSINOPHIL',2:'MONOCYTE',3:'LYMPHOCYTE'}
dict_characters2 = {0:'Mononuclear',1:'Polynuclear'}
import cv2
import scipy
import os
import csv
from tqdm import tqdm
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    z = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 0
                label2 = 1
            elif wbc_type in ['EOSINOPHIL']:
                label = 1
                label2 = 1
            elif wbc_type in ['MONOCYTE']:
                label = 2
                label2 = 0
            elif wbc_type in ['LYMPHOCYTE']:
                label = 3 
                label2 = 0
            
            for image_filename in tqdm(os.listdir(folder + wbc_type)):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (64,64), interpolation = cv2.INTER_AREA)
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
                    z.append(label2)
    X = np.asarray(X)
    y = np.asarray(y)
    z = np.asarray(z)
    return X,y,z
X_train, y_train, z_train = get_data('/content/dataset2-master/dataset2-master/images/TRAIN/')
X_test, y_test, z_test = get_data('/content/dataset2-master/dataset2-master/images/TEST/')


from keras.utils.np_utils import to_categorical
# y_trainHot = to_categorical(y_train, num_classes = 5)
# y_testHot = to_categorical(y_test, num_classes = 5)
# z_trainHot = to_categorical(z_train, num_classes = 2)
# z_testHot = to_categorical(z_test, num_classes = 2)
print(X_train.shape, X_test.shape)

plt.imshow(X_train[0])

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_HEIGHT = 64
IMG_WIDTH = 64

def random_crop(image):
    cropped_image = tf.image.random_crop(
    image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):

    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):

    # image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image):
    image = normalize(image)
    return image

train_zero_idx, train_one_idx = np.where(z_train == 0)[0], np.where(z_train == 1)[0] 
test_zero_idx, test_one_idx = np.where(z_test == 0)[0], np.where(z_test == 0)[0]

train_zero, train_one = X_train[train_zero_idx][:1000], X_train[train_one_idx][:1000]
test_zero, test_one = X_test[test_zero_idx][:1000], X_test[test_one_idx][:1000]

# train_zero, train_one, test_zero, test_one = expand_dimentsion(train_zero), expand_dimentsion(train_one), expand_dimentsion(test_zero), expand_dimentsion(test_one)

train_zero = tf.data.Dataset.from_tensor_slices(train_zero)
train_one = tf.data.Dataset.from_tensor_slices(train_one)
test_zero = tf.data.Dataset.from_tensor_slices(test_zero)
test_one = tf.data.Dataset.from_tensor_slices(test_one)

train_zero = train_zero.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_one = train_one.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zero = test_zero.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_one = test_one.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_zero = next(iter(train_zero))
sample_one = next(iter(train_one))
plt.subplot(121)
plt.title('Mononuclear')
plt.imshow(sample_zero[0] * 0.5 + 0.5)


plt.subplot(122)
plt.title('Mononuclear with random jitter')
plt.imshow(random_jitter(sample_zero[0]) * 0.5 + 0.5)
plt.show()
plt.subplot(121)
plt.title('Polynuclear')
plt.imshow(sample_one[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Polynuclear with random jitter')
plt.imshow(random_jitter(sample_one[0]) * 0.5 + 0.5)
plt.show()

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
import numpy as np
import math

epsilon = 1e-5
def conv(numout,kernel_size=3,strides=1,kernel_regularizer=0.0005,padding='same',use_bias=False,name='conv'):
    return tf.keras.layers.Conv2D(name=name,filters=numout, kernel_size=kernel_size,strides=strides, padding=padding,use_bias=use_bias, kernel_regularizer=l2(kernel_regularizer),kernel_initializer=tf.random_normal_initializer(stddev=0.1))

	
def convt(numout,kernel_size=3,strides=1,kernel_regularizer=0.0005,padding='same',use_bias=False,name='conv'):
    return tf.keras.layers.Conv2DTranspose(name=name,filters=numout, kernel_size=kernel_size,strides=strides, padding=padding,use_bias=use_bias, kernel_regularizer=l2(kernel_regularizer),kernel_initializer=tf.random_normal_initializer(stddev=0.1))	
def bn(name,momentum=0.9):
    return tf.keras.layers.BatchNormalization(name=name,momentum=momentum)


class c7s1_k(keras.Model):
    def __init__(self,scope: str="c7s1_k",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(c7s1_k, self).__init__(name=scope)
        self.conv1 = conv(numout=k,kernel_size=7,kernel_regularizer=reg,padding='valid',name='conv')
        self.norm =norm
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False,activation='Relu'):
        x = tf.pad(x, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        if activation is 'Relu':
            x = tf.nn.relu(x)
        else:
            x = tf.nn.tanh(x)
        return x  

class dk(keras.Model):
    def __init__(self,scope: str="dk",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(dk, self).__init__(name=scope)
        self.norm =norm
        self.conv1 = conv(numout=k,kernel_size=3,strides=[2, 2],kernel_regularizer=reg,padding='same',name='conv')
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False):
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        return x 

		
class Rk(keras.Model):
    def __init__(self,scope: str="Rk",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(Rk, self).__init__(name=scope)
        self.norm =norm
        self.conv1 = conv(numout=k,kernel_size=3,kernel_regularizer=reg,padding='valid',name='layer1/conv')
        if norm is 'instance':
            self.scale1 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer1/scale')
            self.offset1 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer1/offset')
            self.scale2 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer2/scale')
            self.offset2 = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='layer2/offset')
        elif norm is 'bn':
            self.bn1 = bn(name='layer1/bn')
            self.bn2 = bn(name='layer2/bn')
        self.conv2 = conv(numout=k,kernel_size=3,kernel_regularizer=reg,padding='valid',name='layer2/conv')

    def call(self,x,training=False):
        inputs = x
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale1 * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset1
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        x = self.conv2(x)		
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale2 * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset2
        elif self.norm is 'bn':
            x = self.bn2(x,training=training)		
        return x + inputs 		
		
		
	
		
class n_res_blocks(keras.Model):
    def __init__(self,scope: str="n_res_blocks",n:int =6,k:int=16,reg:float=0.0005,norm:str="instance"):
        super(n_res_blocks, self).__init__(name=scope)
        self.group=[]
        self.norm =norm
        for i in range(n):
            self.group.append(Rk(scope='Rk_'+str(i+1),k=k,reg=reg,norm=norm))
    def call(self,x,training=False):
        for i in range(len(self.group)):
            x = self.group[i](x,training=training)
        return x 
		
class uk(keras.Model):
    def __init__(self,scope: str="uk",k:int =16,reg:float=0.0005,norm:str="instance"):
        super(uk, self).__init__(name=scope)
        self.norm =norm
        #self.conv1 = conv(numout=k,kernel_size=3,kernel_regularizer=reg,padding='valid',name='conv')
        self.conv1 = convt(numout=k,kernel_size=3,strides=[ 2 , 2 ],kernel_regularizer=reg,padding='same',name='conv')
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False):
        #height = x.shape[1]
        #width = x.shape[2]
        #x=tf.compat.v1.image.resize_images(x, [2*height,2*width],method = 0, align_corners = True)
        #x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        return x 	        
		
class Ck(keras.Model):
    def __init__(self,scope: str="uk",k:int =16,stride:int=2,reg:float=0.0005,norm:str="instance"):
        super(Ck, self).__init__(name=scope)
        self.norm =norm
        self.conv1 = conv(numout=k,kernel_size=3,strides=[ stride, stride],kernel_regularizer=reg,padding='same',name='conv')
        if norm is 'instance':
            self.scale = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='scale')
            self.offset = tf.Variable(initial_value =tf.random_normal_initializer(stddev=0.1)(shape=[k]),name='offset')
        elif norm is 'bn':
            self.bn1 = bn(name='bn')
    def call(self,x,training=False,slope=0.2):
        x = self.conv1(x)
        if self.norm is 'instance':
            mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
            x = self.scale * tf.math.divide(x - mean, tf.math.sqrt(var + epsilon)) + self.offset
        elif self.norm is 'bn':
            x = self.bn1(x,training=training)
        x = tf.nn.leaky_relu(x,slope)
        return x 

class last_conv(keras.Model):
    def __init__(self,scope: str="last_conv",reg:float=0.0005):
        super(last_conv, self).__init__(name=scope)
        self.conv1 = conv(numout=1,kernel_size=4,kernel_regularizer=reg,padding='same',name='conv')
    def call(self,x,use_sigmoid=False):
        x = self.conv1(x)
        if use_sigmoid:
            output = tf.nn.sigmoid(x)
        return x

import tensorflow as tf
import tensorflow.keras as keras
class Discriminator(keras.Model):
    def __init__(self,scope: str="Discriminator",reg:float=0.0005,norm:str="instance"):
        super(Discriminator, self).__init__(name=scope)
        self.ck1 = Ck(scope="C64",k=64,reg=reg,norm=norm)
        self.ck2 = Ck(scope="C128",k=128,reg=reg,norm=norm)
        self.ck3 = Ck(scope="C256",k=256,reg=reg,norm=norm)
        self.ck4 = Ck(scope="C512",k=512,reg=reg,norm=norm)
        self.last_conv = last_conv(scope="output",reg=reg)
    def call(self,x,training=False,use_sigmoid=False,slope=0.2):
        x=self.ck1(x,training=training,slope=slope)
        x=self.ck2(x,training=training,slope=slope)
        x=self.ck3(x,training=training,slope=slope)
        x=self.ck4(x,training=training,slope=slope)
        x=self.last_conv(x,use_sigmoid=use_sigmoid)
        return x


class Generator(keras.Model):
    def __init__(self,scope: str="Generator",ngf:int=64,reg:float=0.0005,norm:str="instance",more:bool=True):
        super(Generator, self).__init__(name=scope)
        self.c7s1_32=c7s1_k(scope="c7s1_32",k=ngf,reg=reg,norm=norm)
        self.d64 = dk(scope="d64",k=2*ngf,reg=reg,norm=norm)
        self.d128 = dk(scope="d128",k=4*ngf,reg=reg,norm=norm) 
        if more:
            self.res_output=n_res_blocks(scope="8_res_blocks",n=8,k=4*ngf,reg=reg,norm=norm)
        else:
            self.res_output=n_res_blocks(scope="6_res_blocks",n=6,k=4*ngf,reg=reg,norm=norm)
        self.u64=uk(scope="u64",k=2*ngf,reg=reg,norm=norm)
        self.u32=uk(scope="u32",k=ngf,reg=reg,norm=norm)
        self.outconv = c7s1_k(scope="output",k=3,reg=reg,norm=norm)
    def call(self,x,training=False):
        x = self.c7s1_32(x,training=training,activation='Relu')
        x = self.d64(x,training=training)
        x = self.d128(x,training=training)
        x = self.res_output(x,training=training)
        x = self.u64(x,training=training)
        x = self.u32(x,training=training)
        x = self.outconv(x,training=training,activation='tanh')
        return x

generator_g = Generator('G')
generator_f = Generator('F')

discriminator_x = Discriminator('X')
discriminator_y = Discriminator('Y')

"""## Loss functions"""

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

"""Cycle consistency means the result should be close to the original input. For example, if one translates a sentence from English to French, and then translates it back from French to English, then the resulting sentence should be the same as the  original sentence.

In cycle consistency loss, 

* Image $X$ is passed via generator $G$ that yields generated image $\hat{Y}$.
* Generated image $\hat{Y}$ is passed via generator $F$ that yields cycled image $\hat{X}$.
* Mean absolute error is calculated between $X$ and $\hat{X}$.

$$forward\ cycle\ consistency\ loss: X -> G(X) -> F(G(X)) \sim \hat{X}$$

$$backward\ cycle\ consistency\ loss: Y -> F(Y) -> G(F(Y)) \sim \hat{Y}$$


![Cycle loss](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/cycle_loss.png?raw=1)
"""

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

"""As shown above, generator $G$ is responsible for translating image $X$ to image $Y$. Identity loss says that, if you fed image $Y$ to generator $G$, it should yield the real image $Y$ or something close to image $Y$.

If you run the zebra-to-horse model on a horse or the horse-to-zebra model on a zebra, it should not modify the image much since the image already contains the target class.

$$Identity\ loss = |G(Y) - Y| + |F(X) - X|$$
"""

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

"""Initialize the optimizers for all the generators and the discriminators."""

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

"""## Checkpoints"""

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

"""## Training

Note: This example model is trained for fewer epochs (40) than the paper (200) to keep training time reasonable for this tutorial. Predictions may be less accurate. 
"""

EPOCHS = 40

def generate_images(model, test_input):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

"""Even though the training loop looks complicated, it consists of four basic steps:

* Get the predictions.
* Calculate the loss.
* Calculate the gradients using backpropagation.
* Apply the gradients to the optimizer.
"""

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_zero, train_one)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample one) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_one)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

"""## Generate using test dataset"""

# Run the trained model on the test dataset
for inp in test_ones.take(5):
  generate_images(generator_g, inp)
