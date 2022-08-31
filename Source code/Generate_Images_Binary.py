# Generate images for binary case
# Author Yuchong Yao
import numpy as np

# will generate (n_classes * n_classes * steps) = (4 * steps) images, (2 * steps) for each class
# for exapmple you want to generate samples with the same size as MNIST, pass the output_dim = [28, 28]
 

def generate_images_binary(n_classes, latent_dim, x_test, generator, steps, output_dim, **kwargs):
    np.random.seed(42)
    majority = []
    minority = []
    x_real = x_test * 0.5 + 0.5

    for i in range(steps):
        for c in range(n_classes):
            latent_gen = np.random.normal(size=(n_classes, latent_dim))
            decoded_imgs = generator.predict([latent_gen, np.ones(n)*c])
            decoded_imgs = decoded_imgs * 0.5 + 0.5
            decoded_imgs = tf.image.resize(decoded_imgs, output_dim).numpy()

            if c % 2:
                minority += list(decoded_imgs)
            else:
                majority += list(decoded_imgs)
    return np.array(majority), np.array(minority)
