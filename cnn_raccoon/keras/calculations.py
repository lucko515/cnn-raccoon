import os
import cv2
import datetime
import numpy as np

from tensorflow.keras.preprocessing.image import save_img, smart_resize

from cnn_raccoon.keras.grad_cam import grad_cam_helper
from cnn_raccoon import weights_results, grad_cam_results
from cnn_raccoon import images_top_dir, img_relative_path


def weights_inspector(model, images, output_names):
    weights_top_dir = images_top_dir + "/weights"

    if not os.path.exists(weights_top_dir):
        os.mkdir(weights_top_dir)

    outputs = model.predict(images)

    output_layers = model.outputs
    for l in range(len(output_layers)):
        name = output_names[l]
        layer_dir = weights_top_dir + "/{}".format(name)

        if not os.path.exists(layer_dir):
            os.mkdir(layer_dir)

        for img_id in range(len(images)):
            image_dir = layer_dir + "/img_{}".format(img_id)

            if not os.path.exists(image_dir):
                os.mkdir(image_dir)

            img_layer_weights = outputs[l][img_id]
            if img_layer_weights.shape[0] > 4 and img_layer_weights.shape[1] > 4:
                for f_num in range(img_layer_weights.shape[-1]):

                    ts = datetime.datetime.now().timestamp()

                    _path = image_dir + '/filter_{}_{}.jpg'.format(str(ts).replace(".", ""), f_num)
                    weights_relative = img_relative_path + "/weights" + "/{}".format(name) + "/img_{}".format(
                        img_id) + '/filter_{}_{}.jpg'.format(str(ts).replace(".", ""), f_num)

                    image = smart_resize(np.expand_dims(img_layer_weights[:, :, f_num], axis=-1), size=(128, 128))

                    save_img(_path, image)

                    if name not in weights_results.keys():
                        weights_results[name] = {}

                    if img_id in weights_results[name].keys():
                        weights_results[name][img_id].append(weights_relative)
                    else:
                        weights_results[name][img_id] = [weights_relative]


def grad_cam_inspector(model, images, grad_cam_classes):
    grad_cam_top_dir = images_top_dir + "/grad_cam"
    if not os.path.exists(grad_cam_top_dir):
        os.mkdir(grad_cam_top_dir)

    for img_id in range(len(images)):

        image_dir = grad_cam_top_dir + "/img_{}".format(img_id)

        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        img = images[img_id]

        for _class in grad_cam_classes:

            out = grad_cam_helper(model, img, _class)

            ts = datetime.datetime.now().timestamp()

            _path = image_dir + '/grad_cam_class_{}_{}.jpg'.format(str(ts).replace(".", ""), _class)
            grad_cam_relative = img_relative_path + "/grad_cam" + "/img_{}".format(img_id) \
                                + '/grad_cam_class_{}_{}.jpg'.format(str(ts).replace(".", ""), _class)

            out = smart_resize(np.expand_dims(out, axis=-1), size=(128, 128))

            image = smart_resize(img, size=(128, 128))

            cam = cv2.applyColorMap(np.uint8(255 * out), cv2.COLORMAP_JET)
            cam = np.float32(cam) + np.float32(image)
            cam = 255 * cam / np.max(cam)

            save_img(_path, np.squeeze(cam))

            if img_id not in grad_cam_results.keys():
                grad_cam_results[img_id] = [grad_cam_relative]
            else:
                grad_cam_results[img_id].append(grad_cam_relative)
