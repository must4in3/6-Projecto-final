import tensorflow as tf
from IPython import display

OUTPUT_CHANNELS = 3
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def downsample(filters, size, apply_batchnorm=True):
    '''
    Function that becomes part of the father function called Generator(). In this function a neuron is defined for the convolution2D.
    Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU).
        1) Conv -> application of various masks to an image
        2) Batchnorm ->  normalize the network inputs to each sub-layer of the network to avoid the 
           negative phenomenon of shift covariates.
        3) Leaky ReLU -> In the context of artificial neural networks, the ReLU is an activation function
    '''
    #Initializer that generates tensors with a normal distribution.
    initializer = tf.random_normal_initializer(0., 0.02)
    # Sequential groups a linear stack of layers into a tf.keras.Model.
    result = tf.keras.Sequential()
    # añado una capa de convolucíon 2D convolution layer (e.g. spatial convolution over images).
    # params filters:Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    # size? existe kernel_size
    # 	An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. 
    # Can be a single integer to specify the same value for all spatial dimensions.
    # strides: i passi della convoluzione lungo l'altezza e la larghezza. 
    # padding: one of "valid" or "same" (case-insensitive).
    # kernel_initializer: Initializer for the kernel weights matrix ( see keras.initializers).
    # use_bias = Boolean, whether the layer uses a bias vector.
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        # BatchNormalization = Normalize and scale inputs or activations. 
        result.add(tf.keras.layers.BatchNormalization())
        # LeakyReLU = Leaky version of a Rectified Linear Unit.
        result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    '''
    Function that becomes part of the father function called Generator(). 
    In this function a neuron is defined for the Transpose convolution2D.
    Parameters equal to the previous function.
    Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU).
        1) Transposed Conv -> When we transpose it convolution, we change the order of the dimension in this matrix convolution,
           which has some differents effects compared to regular convolutions.
        2) Batchnorm ->  normalize the network inputs to each sub-layer of the network 
           to avoid the negative phenomenon of shift covariates.
        3) Dropout -> For each epoch of training choose (randomly) which neurons to keep and which to discard.

    '''
    #Initializer that generates tensors with a normal distribution.
    initializer = tf.random_normal_initializer(0., 0.02)
    # Sequential groups a linear stack of layers into a tf.keras.Model.
    result = tf.keras.Sequential()
        # añado una capa de convolucíon 2D traspuesta (e.g. spatial convolution over images).
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    # BatchNormalization = Normalize and scale inputs or activations
    result.add(tf.keras.layers.BatchNormalization())
    # Applies Dropout to the input.
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    # Rectified Linear Unit activation function.
    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    '''
    This function defines the architecture of our Generator.
    The architecture of generator is a modified U-Net.
        Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
        Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
        There are skip connections between the encoder and decoder (as in U-Net).
    '''
    # Input() is used to instantiate a Keras tensor.
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    # signa el valor de las dimensiones de output entre parentesis
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]
    #Initializer that generates tensors with a normal distribution.
    initializer = tf.random_normal_initializer(0., 0.02)
    # salida
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

  # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    '''
    The function is used in the training step and allow you to create lost object.
    Generator loss It is a sigmoid cross entropy loss of the generated images and an array of ones. 
    The paper also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image. 
    This allows the generated image to become structurally similar to the target image. 
    The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. 
    This value was decided by the authors of the Pix2Pix.
    '''
    # la funcíon se utiliza en el training step
    # crear objecto perdida
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # perdida calcolada entre target y output
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss