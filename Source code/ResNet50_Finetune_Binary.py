import numpy as np 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

"""Load dataset"""

# (X_train, Y_train) , (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
# (X_train, Y_train) , (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
(X_train, Y_train) , (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

"""select classes"""

# this is the example for cifar
train_major_idx = np.where(Y_train == 8)[0] 
train_minor_idx = np.where(Y_train == 2)[0]
train_idx = np.concatenate([train_major_idx, train_minor_idx])
np.random.shuffle(train_idx)
print("train index size: ", train_major_idx.shape, train_minor_idx.shape, train_idx.shape)

test_major_idx = np.where(Y_test == 8)[0]
test_minor_idx = np.where(Y_test == 2)[0]
test_idx = np.concatenate([test_major_idx, test_minor_idx])
np.random.shuffle(test_idx)
print("test index size: ", test_major_idx.shape, test_minor_idx.shape, test_idx.shape)

X_train, Y_train = X_train[train_idx], Y_train[train_idx]
# Y_train[Y_train == 1] = 2
# Y_train[Y_train == 0] = 1
# Y_train[Y_train == 2] = 0
Y_train[Y_train == 8] = 0
Y_train[Y_train == 2] = 1
X_test, Y_test = X_test[test_idx], Y_test[test_idx]
# Y_test[Y_test == 1] = 2
# Y_test[Y_test == 0] = 1
# Y_test[Y_test == 2] = 0
Y_test[Y_test == 8] = 0
Y_test[Y_test == 2] = 1
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print("train set shape: ", X_train.shape, Y_train.shape)
print("test set shape: ", X_test.shape, Y_test.shape)

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
    x = tf.keras.layers.Dense(2, activation="sigmoid", name="classification")(x)
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
                loss='binary_crossentropy',
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

model.save("ResNet50-Cifar10-binary.h5")

"""load model"""

path = "/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Models/ResNet50/2-class/ResNet50-Cifar10-binary.h5"
load_model = tf.keras.models.load_model(path)
loss, accuracy = load_model.evaluate(X_test, Y_test, batch_size=64)

"""load generated images"""

def process_generate_images(majority_path=None, minority_path=None, greyscale=False, flip=False, **kwargs):
    # load images
    X_majority = np.load(majority_path)
    X_minority = np.load(minority_path)
    
    # scale tp 0 - 255
    X_majority = X_majority * 255. 
    X_minority = X_minority * 255. 

    # expend dimension for grayscale images
    if greyscale:
        X_majority = expand_dimentsion(X_majority) 
        X_minority = expand_dimentsion(X_minority)
    
    # preprocess images
    X_majority = preprocess_image(X_majority)
    X_minority = preprocess_image(X_minority)
    
    # assign labels
    label_majority, label_minority = 0, 1
    if flip:
        label_majority, label_minority = 1, 0
    Y_majority = np.array([[label_majority]]*5000)
    Y_minority = np.array([[label_minority]]*5000)

    # form dataset
    X_generate = np.concatenate([X_majority, X_minority])
    Y_generate = np.concatenate([Y_majority, Y_minority])

    # shuffle dataset
    shuffler = np.random.permutation(X_generate.shape[0])
    X_generate, Y_generate = X_generate[shuffler], Y_generate[shuffler]
    Y_true = Y_generate
    Y_generate = to_categorical(Y_generate)
    print(X_generate.shape, Y_generate.shape, Y_true.shape)

    return X_generate, Y_generate, Y_true

path1 = "/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Bagan_gp experiments/CIFAR10/Cifar-100-Majority.npy"
path2 = "/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Bagan_gp experiments/CIFAR10/Cifar-100-Minority.npy"
X_generate, Y_generate, Y_true = process_generate_images(majority_path=path1, minority_path=path2, greyscale=False, flip=False)
loss, accuracy = load_model.evaluate(X_generate, Y_generate, batch_size=64)
Y_pred = np.argmax(load_model.predict(X_generate), axis=1)
print(tf.math.confusion_matrix(Y_true, Y_pred))
