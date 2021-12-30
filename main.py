#!/usr/bin/env python3

import glob
import os
import time
from typing import List

import tensorflow as tf
import tensorflow_hub as hub

from style_transfer.style_content import StyleContentModel, train_step
from style_transfer.utils import load_img, tensor_to_image
from style_transfer.style_content import vgg_layers


os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# Parameters

style_weight = 1
content_weight = 1000

n_epochs = 10
steps_per_epoch = 10

# Optimizer

learning_rate = 0.02
beta_1 = 0.99
epsilon = 1e-1

# Model

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Files

file_types = ["jpg", "jpeg", "gif", "png", "tiff"]

content_image_paths: List[str] = glob.glob("./images/content/*")
style_image_paths: List[str] = glob.glob("./images/style/*")

def get_content_and_style_layers_vgg19():

    """
    VGG19 has the following layers:

    input_2
    block1_conv1
    block1_conv2
    block1_pool
    block2_conv1
    block2_conv2
    block2_pool
    block3_conv1
    block3_conv2
    block3_conv3
    block3_conv4
    block3_pool
    block4_conv1
    block4_conv2
    block4_conv3
    block4_conv4
    block4_pool
    block5_conv1
    block5_conv2
    block5_conv3
    block5_conv4
    block5_pool

    The lower layers encode lower level visual details like lines and circles etc... and the higher layers encode for
    higher level features that lower level features compose.

    Returns:

    """

    # The highest level concepts
    content_layers = ['block5_conv4']

    # Set to lower-level style features
    style_layers = ['block1_conv1',
                    'block1_conv2',
                    'block2_conv1',
                    'block2_conv2',
                    'block3_conv1']

    # TODO: have a random feature picker?
    # TODO: implement a nice interface to set these hyperparameters
    return content_layers, style_layers


def define_content_and_style_representations_vgg19(content_image):
    """
    Use the intermediate layers of the model to get the content and style representations of the image. Starting from
    the network's input layer, the first few layer activations represent low-level features like edges and textures.
    As you step through the network, the final few layers represent higher-level features—object parts like wheels or
    eyes. In this case, you are using the VGG19 network architecture, a pretrained image classification network.
    These intermediate layers are necessary to define the representation of content and style from the images. For an
    input image, try to match the corresponding style and content target representations at these intermediate
    layers.

    Returns: a List of the top 5 predictions, in tuple format, (class, probability). Best to print the results.

    """
    x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)
    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    return [(class_name, prob) for (number, class_name, prob) in predicted_top_5]


def do_style_transfer_simple(content_image_path: str, style_image_path: str):
    """Do style transfer using a tensorflow hub model.

    Demonstrates the original style-transfer algorithm, which optimizes the image content to a particular style.

    Args:
        content_image_path (str): Path to content image.
        style_image_path (str): Path to style image.

    Returns:
        object: ?
    """
    content_image, style_image = load_img(content_image_path), load_img(style_image_path)
    return hub_model(tf.constant(content_image), tf.constant(style_image))[0]


def print_style_model_stats_vgg19(style_layers, style_image):
    """Look at the statistics of each layer's output."""

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    for name, output in zip(style_layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()


def print_content_model_stats_vgg19(content_layers, content_image):
    """Look at the statistics of each layer's output."""

    content_extractor = vgg_layers(content_layers)
    content_outputs = content_extractor(content_image * 255)

    for name, output in zip(content_layers, content_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()


def do_style_transfer(n_epochs, steps_per_epoch, content_image_path: str, style_image_path: str, optimizer):
    """

    Args:
        optimizer:
        n_epochs:
        steps_per_epoch:
        content_image_path:
        style_image_path:
    """

    # 1. Load images from path
    content_image, style_image = load_img(content_image_path), load_img(style_image_path)

    # 2. Get content and style layers, TODO: ability to adjust
    content_layers, style_layers = get_content_and_style_layers_vgg19()

    # 3. Create the StyleContentModel
    extractor = StyleContentModel(style_layers,
                                  content_layers,
                                  style_weight,
                                  content_weight)

    extractor.set_style_targets(style_image)
    extractor.set_content_targets(content_image)

    # 4. Initialize the Image Variable
    image = None
    image = tf.Variable(content_image)

    # 5. Optimize Model
    start = time.time()
    step = 0
    for n in range(n_epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(extractor, image, optimizer)
            print(".", end='', flush=True)
        print("Train step: {}".format(step))
    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    # 6. Save Image
    content_name = content_image_path.split('/')[-1]
    style_name = style_image_path.split('/')[-1]
    file_name = f'{content_name}-{style_name}.png'
    tensor_to_image(image).save(f'./results/{file_name}')


if __name__ == '__main__':
    """
    Here we'll want to iterate over all style files and content files.
    """

    print(f'You have {len(content_image_paths)} content images.')
    print(f'You have {len(style_image_paths)} style images.')

    if not os.path.isdir('./results/'):
        os.mkdir('./results/')

    for i, c_image_path in enumerate(content_image_paths):
        for j, s_image_path in enumerate(style_image_paths):

            file_name = f"{c_image_path.split('/')[-1]}-{s_image_path.split('/')[-1]}.png"

            if not os.path.isfile(f'./results/{file_name}'):

                s_image_filetype = s_image_path.split(".")[-1]
                c_image_filetype = c_image_path.split(".")[-1]

                if c_image_filetype in file_types and s_image_filetype in file_types:
                    print(f'Now working on content image {i}, style image {j}, filename: {file_name}...')
                    # TODO: for each call, the resulting image should also have a corresponding json metadata file.
                    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon)
                    do_style_transfer(n_epochs=n_epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      content_image_path=c_image_path,
                                      style_image_path=s_image_path,
                                      optimizer=optimizer)
                else:
                    print("Problem with file types:")
                    print(c_image_path)
                    print(s_image_path)

    """
    Here we will most likely want to print out an image that shows the original content image plus the style image plus the 
    resulting image. This can be included in the NFT as a treat.
    """
