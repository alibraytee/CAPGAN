# Author Yuanbang Ma 
# SSIM score is in (-1, 1]
def calculate_ssim(real_majority, real_minority, gen_majority, gen_minority, steps = 5):
    
    # resize the number of real majority and minority
    # make sure the size of real image and generated image are same
    real_maj_idx = np.random.choice(real_majority.shape[0], gen_majority.shape[0])
    real_majority = real_majority[real_maj_idx].astype('float32') / 255.
    
    real_min_idx = np.random.choice(real_minority.shape[0], gen_minority.shape[0])
    real_minority = real_minority[real_min_idx].astype('float32') / 255.
    
    
    majority_score = 0
    minority_score = 0
    
    # multiple calculation of the diffrent real images and generated images
    for i in range(steps):
        # random the generated images and real images for majority
        np.random.shuffle(real_majority)
        np.random.shuffle(gen_majority)
        
        # calculate the SSIM of Majority
        maj_ssim = tf.image.ssim(real_majority, gen_majority, max_val=1.0)
        majority_score += np.average(maj_ssim.numpy())
        
        # random the generated images and real images for minority
        np.random.shuffle(real_minority)
        np.random.shuffle(gen_minority)
        
        # calculate the SSIM of Minority
        min_ssim = tf.image.ssim(real_minority, gen_minority, max_val=1.0)
        minority_score +=np.average(min_ssim.numpy())
    
    majority_score /= steps
    minority_score /= steps
    
    print('>>SSIM(0): %.3f' % majority_score)
    print('>>SSIM(1): %.3f' % minority_score)
    
    return majority_score, minority_score



###----------------------------------- ##
# Please choice your training dataset  ##
###----------------------------------- ##
# # Load MNIST Fashion
from tensorflow.keras.datasets.fashion_mnist import load_data

# # Load MNIST
#from tensorflow.keras.datasets.mnist import load_data

# # Load CIFAR-10
#from tensorflow.keras.datasets.cifar10 import load_data

# # Load training set
(images, labels), (_,_) = load_data()
images = change_image_shape(images)
labels = labels.reshape(-1)

# # select the real majority and real minority
###----------------------------------- ##
#    Don't forget choice the label     ##
###----------------------------------- ##
real_majority = images[labels ==1]
real_minority = images[labels ==0]
print(real_majority.shape, real_minority.shape)


# # Load generated image data
###----------------------------------- ##
#  Please load your generated dataset  ##
###----------------------------------- ##
gen_majority = np.load('fashion_1_0_rate10_majority.npy')
gen_minority = np.load('fashion_1_0_rate10_minority.npy')
print(gen_majority.shape,gen_minority.shape)
print()

## measure SSIM score
majority_score, minority_score = calculate_ssim(real_majority,real_minority,gen_majority ,gen_minority,steps = 10)
print()
print('The SSIM score for Majority: %.3f' % majority_score)
print('The SSIM score for Minority: %.3f' % minority_score)
