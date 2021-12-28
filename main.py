#!/usr/bin/env python3

import glob
import os
import time

import tensorflow as tf
import tensorflow_hub as hub

from style_transfer.style_content_model import StyleContentModel
from style_transfer.utils import load_img, tensor_to_image, vgg_layers, clip_0_1, high_pass_x_y


# Parameters

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

epochs = 10
steps_per_epoch = 100

# Optimizer

learning_rate = 0.02
beta_1 = 0.99
epsilon = 1e-1

opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon)

# Model

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Files

content_image_paths = glob.glob("./images/content/*")
style_image_paths = glob.glob("./images/style/*")


def style(content_image_path, style_image_path, optimizer):

    content_image, style_image = load_img(content_image_path), load_img(style_image_path)

    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    x: object = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    
    vgg: object = tf.keras.applications.VGG19(include_top=True, weights='imagenet')

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255)

    extractor = StyleContentModel(style_layers,
                                  content_layers,
                                  style_weight,
                                  content_weight,
                                  total_variation_weight,
                                  optimizer)

    results = extractor(tf.constant(content_image))

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)


    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            extractor.train_step(image)
            print(".", end='', flush=True)
    print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    x_deltas, y_deltas = high_pass_x_y(content_image)
    x_deltas, y_deltas = high_pass_x_y(image)

    sobel = tf.image.sobel_edges(content_image)

    extractor.total_variation_loss(image).numpy()

    tf.image.total_variation(image).numpy()

    image = tf.Variable(content_image)

    # Run the Optimization

    start = time.time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            extractor.train_step(image)
            print(".", end='', flush=True)
        print("Train step: {}".format(step))

    end = time.time()

    print("Total time: {:.1f}".format(end - start))

    content_name = content_image_path.split('/')[-1]
    style_name = style_image_path.split('/')[-1]

    file_name = f'{content_name}-{style_name}.png'

    tensor_to_image(image).save(file_name)

# TODO: need to download the file here.


if __name__ == '__main__':
    """
    Here we'll want to iterate over all style files and content files.
    """

    if not os.path.isdir('./Results/'):
        os.mkdir('./Results/')

    for c_image_path, s_image_path in zip(content_image_paths, style_image_paths):
        style(c_image_path, s_image_path, opt)
    """
    Here we will most likely want to print out an image that shows the original content image plus the style image plus the 
    resulting image. This can be included in the NFT as a treat.
    """
