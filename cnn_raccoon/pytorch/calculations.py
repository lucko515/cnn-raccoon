import os
import cv2
import numpy as np

import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F

from cnn_raccoon import weights_results, layers_dict, grad_cam_results, saliency_map_results
from cnn_raccoon import images_top_dir, img_relative_path


def weights_inspector(weights, images):
    weights_top_dir = images_top_dir + "/weights"

    if not os.path.exists(weights_top_dir):
        os.mkdir(weights_top_dir)

    for layer in layers_dict.keys():

        # Because of memory explosion only Conv layers are inspected
        if isinstance(layers_dict[layer], nn.Conv2d) \
                or isinstance(layers_dict[layer], nn.MaxPool2d):
            layer_dir = weights_top_dir + "/{}".format(layer)
            if not os.path.exists(layer_dir):
                os.mkdir(layer_dir)

            for img_id in range(len(images)):
                image_dir = layer_dir + "/img_{}".format(img_id)

                if not os.path.exists(image_dir):
                    os.mkdir(image_dir)

                img = images[img_id]
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                else:
                    img = img

                layer_obj = layers_dict[layer]
                out, _ = weights(img, layer_obj)

                for f_num in range(out.shape[0]):
                    if out.shape[-1] > 3 and out.shape[-2] > 3:
                        _path = image_dir + '/filter_{}.jpg'.format(f_num)
                        weights_relative = img_relative_path + "/weights" + "/{}".format(layer) + "/img_{}".format(
                            img_id) + '/filter_{}.jpg'.format(f_num)

                        _out = F.interpolate(out[f_num].unsqueeze(dim=1), size=128)

                        save_image(_out, _path)

                        if layer not in weights_results.keys():
                            weights_results[layer] = {}

                        if img_id in weights_results[layer].keys():
                            weights_results[layer][img_id].append(weights_relative)
                        else:
                            weights_results[layer][img_id] = [weights_relative]




def grad_cam_inspector(grad_cam, images, grad_cam_classes):
    grad_cam_top_dir = images_top_dir + "/grad_cam"
    if not os.path.exists(grad_cam_top_dir):
        os.mkdir(grad_cam_top_dir)

    for img_id in range(len(images)):

        image_dir = grad_cam_top_dir + "/img_{}".format(img_id)

        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        img = images[img_id]
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        else:
            img = img

        for _class in grad_cam_classes:
            out, _ = grad_cam(img, None, target_class=_class)

            _path = image_dir + '/grad_cam_class_{}.jpg'.format(_class)
            grad_cam_relative = img_relative_path + "/grad_cam" + "/img_{}".format(img_id) \
                                + '/grad_cam_class_{}.jpg'.format(_class)

            out = F.interpolate(out, size=128)
            save_image(out, _path)

            if img_id not in grad_cam_results.keys():
                grad_cam_results[img_id] = [grad_cam_relative]
            else:
                grad_cam_results[img_id].append(grad_cam_relative)


def saliency_map_inspector(saliency_map, images):
    saliency_map_top_dir = images_top_dir + "/saliency_map"
    if not os.path.exists(saliency_map_top_dir):
        os.mkdir(saliency_map_top_dir)

    for layer in layers_dict.keys():
        if isinstance(layers_dict[layer], nn.Conv2d) or \
                isinstance(layers_dict[layer], nn.Linear) or \
                isinstance(layers_dict[layer], nn.MaxPool2d):

            layer_dir = saliency_map_top_dir + "/{}".format(layer)
            if not os.path.exists(layer_dir):
                os.mkdir(layer_dir)

            for img_id in range(len(images)):

                image_dir = layer_dir + "/img_{}".format(img_id)

                if not os.path.exists(image_dir):
                    os.mkdir(image_dir)

                img = images[img_id]
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                else:
                    img = img

                layer_obj = layers_dict[layer]
                out, _ = saliency_map(img, layer_obj)

                out = out.squeeze()

                _path = image_dir + '/saliency_map.jpg'
                saliency_map_relative = img_relative_path + "/saliency_map" + "/{}".format(layer) + "/img_{}".format(
                    img_id) + '/saliency_map.jpg'

                out = F.interpolate(out.unsqueeze(dim=0).unsqueeze(dim=0), size=128)
                save_image(out, _path)

                if layer not in saliency_map_results.keys():
                    saliency_map_results[layer] = {}

                if img_id in saliency_map_results[layer].keys():
                    saliency_map_results[layer][img_id].append(saliency_map_relative)
                else:
                    saliency_map_results[layer][img_id] = [saliency_map_relative]

