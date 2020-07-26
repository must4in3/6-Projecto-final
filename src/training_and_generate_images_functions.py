import tensorflow as tf
from matplotlib import pyplot as plt
from src.generator_functions import Generator
from src.discriminator_functions import Discriminator
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

