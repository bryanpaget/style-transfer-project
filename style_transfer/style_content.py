#!/usr/bin/env python
from typing import Dict

import tensorflow as tf

from style_transfer.utils import high_pass_x_y


class StyleContentModel(tf.keras.models.Model):
    """

    """

    def __init__(self, style_layers, content_layers, style_weight, content_weight):
        super(StyleContentModel, self).__init__()
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.vgg.trainable = False

        self.layer_weights = None  # TODO: implement this to give different weights to style layers
        # probably want exponential decay or arithmetic or something else
        self.style_targets = None
        self.content_targets = None

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    def set_style_targets(self, style_image):
        """

        Args:
            style_image:
        """
        self.style_targets = self.call(style_image)['style']

    def set_content_targets(self, content_image):
        """

        Args:
            content_image:
        """
        self.content_targets = self.call(content_image)['content']

    def style_content_loss(self, outputs, style_targets, content_targets):
        """

        Args:
            outputs:
            style_targets:
            content_targets:

        Returns:

        """
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ])

        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ])

        content_loss *= self.content_weight / self.num_content_layers

        return style_loss + content_loss


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""

    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    return tf.keras.Model([vgg.input], outputs)


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# TODO: make a function that prints a page with content, style and result
def print_content_style_result():
    pass


# TODO: understand tensorflow function decorator better, enabling breaks code
#@tf.function()
def train_step(model, image, optimizer):
    """

    Args:
        model:
        image:
        optimizer:
    """
    with tf.GradientTape() as tape:
        outputs = model(image)
        loss = model.style_content_loss(outputs, model.style_targets, model.content_targets)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def metadata():
    """These metadata will help us engineer images to be more interesting or to fit a certain style."""
    content_style_name: str = ""
    style_file_name: str = ""
    hyperparameters: Dict[str, str] = {}
    # content weight
    # style weight
    # which layers
    # layer weights?
    vgg19_classification_head_predictions = []
    vgg19_style_layers = []
    vgg19_content_lauers = []
