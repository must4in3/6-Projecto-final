import tensorflow as tf
from src.generator_functions import downsample, upsample, generator_loss

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def Discriminator():
    '''
    This function allow you to build the dscriminator.
    The Discriminator is a PatchGAN.
    Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU).
    The shape of the output after the last layer is (batch_size, 30, 30, 1)
    Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
    Discriminator receives 2 inputs.
        1) Input image and the target image, which it should classify as real.
        2) Input image and the generated image (output of generator), which it should classify as fake.
        3) We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
    '''
    #Initializer that generates tensors with a normal distribution.
    initializer = tf.random_normal_initializer(0., 0.02)
    # el shape tiene que ser el mismo de la salida del generador (tamaño de las imagenes)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    # concatenar la imagen de input(parecida a mascara) con la imagen target (palacio original) 
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    # 3 convolucíones 2D
    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
    # This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    # otra convolucíon 2D
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    # BatchNormalization = Normalize and scale inputs or activations. 
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    # Leaky version of a Rectified Linear Unit.
    # It allows a small gradient when the unit is not active:
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    # This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    # ultima convolucíon con una salida de (bs, 30, 30, 1)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    '''
    The discriminator loss function takes 2 inputs; real images, generated images
    real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images)
    generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since these are the fake images)
    Then the total_loss is the sum of real_loss and the generated_loss
    P.S. I call this function in the father function train_step()
    '''
    # real_loss is a sigmoid cross entropy loss of the real images 
    #and an array of ones(since these are the real images)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    # generated_loss is a sigmoid cross entropy loss of the generated 
    #images and an array of zeros(since these are the fake images)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    #  total_loss is the sum of real_loss and the generated_loss
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss