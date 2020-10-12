
import os
import cv2
import torch
import numpy as np

from torchvision.utils import save_image
import torch.nn.functional as F

from cnn_raccoon import input_images_dict
from cnn_raccoon import images_top_dir, img_relative_path


def tensor_image_converter(tensor):
    tensor = tensor.squeeze()

    if len(tensor.shape) > 2:
        tensor = tensor.permute(1, 2, 0)

    img = tensor.detach().cpu().numpy()
    return img


def module2traced(module, inputs):
    """
    Function taken from: https://github.com/FrancescoSaverioZuppichini/A-journey-into-Convolutional-Neural-Network-visualization-


    :param module:
    :param inputs:
    :return:
    """
    handles, modules = [], []

    def trace(module, inputs, outputs):
        modules.append(module)

    def traverse(module):
        for m in module.children():
            traverse(m)
        is_leaf = len(list(module.children())) == 0
        if is_leaf:
            handles.append(module.register_forward_hook(trace))

    traverse(module)

    _ = module(inputs)

    [h.remove() for h in handles]

    return modules


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


def image2cam(image, cam):
    """
    Credits to: https://github.com/FrancescoSaverioZuppichini/A-journey-into-Convolutional-Neural-Network-visualization-

    :param image:
    :param cam:
    :return:
    """
    h, w, c = image.shape
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (w, h))

    cam = np.uint8(cam * 255.0)
    img_with_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img_with_cam = cv2.cvtColor(img_with_cam, cv2.COLOR_BGR2RGB)
    img_with_cam = img_with_cam + (image * 255)
    img_with_cam /= np.max(img_with_cam)

    return img_with_cam


def tensor2cam(image, cam):
    """
    Credits to: https://github.com/FrancescoSaverioZuppichini/A-journey-into-Convolutional-Neural-Network-visualization-

    :param image:
    :param cam:
    :return:
    """
    image_with_heatmap = image2cam(image.squeeze().permute(1, 2, 0).cpu().numpy(), cam.detach().cpu().numpy())

    return torch.from_numpy(image_with_heatmap).permute(2, 0, 1)


def save_input_images(images):

    input_images = images_top_dir + "/input_images"

    if not os.path.exists(input_images):
        os.mkdir(input_images)

    for i in range(images.shape[0]):
        img_path = input_images + "/img_{}.jpg".format(i)
        img_relative = img_relative_path + "/input_images" + "/img_{}.jpg".format(i)

        image = images[i]
        image = F.interpolate(image, size=128)

        save_image(image, img_path)

        if i in input_images_dict.keys():
            input_images_dict[i].append(img_relative)
        else:
            input_images_dict[i] = img_relative
