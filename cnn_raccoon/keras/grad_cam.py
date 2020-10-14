import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D


def grad_cam_helper(model, image, class_id):
    """
    Source : https://keras.io/examples/vision/grad_cam/
    """
    # Find what is the last convolutional layer in the network
    last_conv_layer = None

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            last_conv_layer = layer

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])

    classifier_layers = [GlobalAveragePooling2D(), model.layers[-1]]

    x = classifier_input
    for layer in classifier_layers:
        x = layer(x)

    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(np.expand_dims(image, axis=0))
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_class_channel = preds[:, class_id]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap
