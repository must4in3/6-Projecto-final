import time
from IPython import display
import tensorflow as tf
from matplotlib import pyplot as plt
from src.generator_functions import Generator, generator_loss
from src.discriminator_functions import Discriminator, discriminator_loss
EPOCHS = 150

def generate_images(model, test_input, tar):
    '''
    Write a function to plot some images during training.
    Pass images from the test dataset to the generator. The generator will then translate the input image into the output. 
    Last step is to plot the predictions.
    Note: The training=True is intentional here since we want the batch statistics while running the model on the test dataset. 
    If we use training=False, we will get the accumulated statistics learned from the training dataset (which we don't want).
    model is the generator.    
    '''
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function()
def train_step(input_image, target, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer):
    '''
    The discriminator receives the input_image and the generated image as the first input. 
    The second input is the input_image and the target_image.
    Next, we calculate the generator and the discriminator loss.
    Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) 
    and apply those to the optimizer.
    Then log the losses to TensorBoard.
    '''
    # GradientTape = Record operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds, generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, checkpoint):
    '''
    The actual training loop:
    Iterates over the number of epochs.
    On each epoch it clears the display, and runs generate_images to show it's progress.
    On each epoch it iterates over the training dataset, printing a '.' for each example.
    It saves a checkpoint every 20 epochs.

    This training loop saves logs you can easily view in TensorBoard to monitor the training progress. 
    Working locally you would launch a separate tensorboard process. In a notebook, if you want to monitor with TensorBoard 
    it's easiest to launch the viewer before starting the training.

    '''
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch, generator, discriminator,generator_optimizer, discriminator_optimizer, summary_writer)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
    checkpoint.save(file_prefix = checkpoint_prefix)

