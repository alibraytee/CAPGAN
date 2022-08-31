

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

"""Load dataset""" 

# (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()
# (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.fashion_mnist.load_data()
(X_train, Y_train) , (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

"""Reshape and add channels for MNIST/Fashion-MNIST"""

def expand_dimentsion(data):
    if len(data.shape) ==3:
        data = np.expand_dims(data, axis=-1)
    data = np.repeat(data, 3, axis=-1)
    data = tf.image.resize(data, [32,32]) 
    return data.numpy()

# X_train = expand_dimentsion(X_train)
# X_test = expand_dimentsion(X_test)
# print(X_train.shape, X_test.shape)

"""Process dataset

"""

def preprocess_image(inputs):
  inputs = inputs.astype('float32')
  output = tf.keras.applications.resnet50.preprocess_input(inputs)
  return output
X_train = preprocess_image(X_train)
X_test = preprocess_image(X_test)

"""Build model"""

def pretained_ResNet50(inputs):
  return tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)

def downstream_classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x

def resnet50(inputs):
    inputs = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    resnet = pretained_ResNet50(inputs)
    output = downstream_classifier(resnet)
    return output

def build_model():
  inputs = tf.keras.layers.Input(shape=(32,32,3))
  output = resnet50(inputs) 
  model = tf.keras.Model(inputs=inputs, outputs = output)
  model.compile(optimizer='SGD', 
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy']) 
  return model


model = build_model()
model.summary()

"""Training"""

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0.0003, patience=3, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

history = model.fit(X_train, Y_train, 
                    epochs=10, validation_data = (X_test, Y_test), 
                    batch_size=64, callbacks=[early_stop]
)

"""Evaluation"""

loss, accuracy = model.evaluate(X_test, Y_test, batch_size=64)

"""Save model"""

model.save("ResNet50-Cifar10.h5")

"""load model"""

path = "/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Models/ResNet50/10-class/ResNet50-Cifar10.h5"
load_model = tf.keras.models.load_model(path)
loss, accuracy = load_model.evaluate(X_test, Y_test, batch_size=64)

"""load generated images"""

X_majority = np.load("/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Bagan_gp experiments/CIFAR10/Cifar-100-Majority.npy")

X_majority = X_majority * 255. # range to 0 - 255

# X_majority = expand_dimentsion(X_majority) # use thie line for MNIST and Fashion-MNIST, but not for CIFAR10
X_majority = preprocess_image(X_majority)

pred = load_model.predict(X_majority)
print(pred)

np.argmax(pred,axis=1)
