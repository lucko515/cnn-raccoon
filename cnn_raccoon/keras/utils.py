import os
import datetime
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import save_img, smart_resize

from cnn_raccoon import input_images_dict
from cnn_raccoon import images_top_dir, img_relative_path


def create_weights_model(input_model):
    """
    Helper function to create Keras model for weights extraction.

    :param input_model: The loaded Keras model through Inspector.

    :return: keras.models.Model object optimized to return weights of each Conv and MaxPool layer.
    """
    outputs = []
    output_names = []
    for layer in input_model.layers:
        if isinstance(layer, Conv2D):
            outputs.append(layer.output)
            output_names.append(layer.name)
        if isinstance(layer, MaxPool2D):
            outputs.append(layer.output)
            output_names.append(layer.name)

    assert len(output_names) == len(outputs)

    model = Model(inputs=input_model.inputs, outputs=outputs)

    return model, output_names


def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    credits to https://github.com/utkuozbulak/pytorch-cnn-visualizations
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_input_images(images):
    """
    Saves input images and expose them to the Flask server.

    :param images: Input images taken from Inspector
    """
    input_images = images_top_dir + "/input_images"

    if not os.path.exists(input_images):
        os.mkdir(input_images)

    for i in range(images.shape[0]):
        ts = datetime.datetime.now().timestamp()

        img_path = input_images + "/img_{}_{}.jpg".format(str(ts).replace(".", ""), i)
        img_relative = img_relative_path + "/input_images" + "/img_{}_{}.jpg".format(str(ts).replace(".", ""), i)

        image = images[i]
        image = smart_resize(image, size=(128, 128))

        save_img(img_path, image)

        if i in input_images_dict.keys():
            input_images_dict[i].append(img_relative)
        else:
            input_images_dict[i] = img_relative

