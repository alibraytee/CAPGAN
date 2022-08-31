# Author Yuchong Yao
# Adapted from https://github.com/GH920/improved-bagan-gp/blob/master/codes/fid_score.py
# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
 
# Libraries
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

# utility functions
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

# load real majority and minority
from tensorflow.keras.datasets.fashion_mnist import load_data
(_, _), (X_test, Y_test) = load_data()
major_idx = np.where(Y_test == 7)[0] # majority sneaker
minor_idx = np.where(Y_test == 8)[0] # minority bag
real_majority = X_test[major_idx].astype('float32') / 255.
real_minority = X_test[minor_idx].astype('float32') / 255.

# load generated majority and minority
gen_majority = np.load("/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Bagan_gp experiments/FashionMNIST/Fashion-MNIST50-Majority.npy") 
gen_minority = np.load("/content/drive/MyDrive/CS31-2-Deep learning with class imbalance/Bagan_gp experiments/FashionMNIST/Fashion-MNIST50-Minority.npy") 
gen_majority = gen_majority[:1000]
gen_minority = gen_minority[:1000]

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(224, 224, 3))


def fid(real_majority, real_minority, gen_majority, gen_minority, model, **kwargs):
    # resize 
    real_majority = scale_images(real_majority, (224, 224, 3))
    real_minority = scale_images(real_minority, (224, 224, 3))
    gen_majority = scale_images(gen_majority, (224, 224, 3))
    gen_minority = scale_images(gen_minority, (224, 224, 3))

    # preprocess images
    real_majority = preprocess_input(real_majority)
    real_minority = preprocess_input(real_minority)
    gen_majority = preprocess_input(gen_majority)
    gen_minority = preprocess_input(gen_minority)

    # calculate fid
    fid_majority = calculate_fid(model, gen_majority, real_majority)
    fid_minority = calculate_fid(model, gen_minority, real_minority)

    return fid_majority, fid_minority

fid_majority, fid_minority = fid(real_majority=real_majority, real_minority=real_minority, gen_majority=gen_majority, gen_minority=gen_minority, model=model)
print("FID for Majority: ", fid_majority)
print("FID for Minority: ", fid_minority)
