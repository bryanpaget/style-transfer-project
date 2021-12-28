#!/usr/bin/env python

import tensorflow as tf

from style_transfer.utils import vgg_layers, gram_matrix, clip_0_1, high_pass_x_y


class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers, style_weight, content_weight, total_variation_weight, opt):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight
        self.opt = opt

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

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    def style_content_loss(self, outputs):

        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([
            tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ])

        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([
            tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ])

        content_loss *= self.content_weight / self.num_content_layers

        return style_loss + content_loss

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self(image)
            loss = self.style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    @staticmethod
    def total_variation_loss(image):
        x_deltas, y_deltas = high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))