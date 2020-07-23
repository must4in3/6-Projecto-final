import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
from IPython import display

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    '''
    This function receives a path string, and returns the input image and the ground through
    '''
    #Reads and outputs the entire contents of the input filename. Return a Tensor of type string.
    image = tf.io.read_file(image_file)
    # Decodificar image. The attr channels indicates the desired number of color channels for the decoded image.
    image = tf.image.decode_jpeg(image)
    # 512 pixel 256*2
    w = tf.shape(image)[1]
    # 512/2 pixel = 256, para partir la foto de la mascara
    w = w // 2
    # separar real imagen desde la imagen de input
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]
    # Casts a tensor to a new type.
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    '''
    resize las dos imagenes con el metodo de los NEAREST_NEIGHBOR
    '''
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    '''
    this function randomly crops a tensor to a given size.
    '''
    #Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    stacked_image = tf.stack([input_image, real_image], axis=0)

    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    '''
    normalizing the images to [-1, 1]
    '''
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    '''
    This function allow you to:
    1) resizing the images to 286 x 286 x 3
    2) randomly cropping to 256 x 256 x 3
    3) random mirroring
    '''
    input_image, real_image = resize(input_image, real_image, 286, 286)

    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    '''
    This function automate the whole process of loading train images, recalling the functions previously created.
    '''
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    '''
    This function automate the whole process of loading test images, recalling the functions previously created.
    '''
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image