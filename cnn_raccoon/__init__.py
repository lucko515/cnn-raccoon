import os
import json
import shutil
import inspect

import numpy as np

lib_path = os.path.dirname(os.path.abspath(__file__))
images_top_dir = lib_path + "/static/images/project"
img_relative_path = "/images/project"

# Third party libraries
from flask import Flask, redirect, url_for, render_template

# Create Flask application
app = Flask(__name__)

# Global variables
prediction_model = None
nodes = []
edges = []
layers_dict = {}
weights_results = {}
grad_cam_results = {}
saliency_map_results = {}
input_images_dict = {}

# Make sure to create image-project folder for the current session
if os.path.exists(images_top_dir):
    shutil.rmtree(images_top_dir, ignore_errors=True)
    os.mkdir(images_top_dir)
else:
    os.mkdir(images_top_dir)


# Home/login screen
@app.route("/", methods=['GET'])
def index():

    return render_template("dashboard.html",
                           nodes=json.dumps(nodes), edges=json.dumps(edges),
                           input_images=json.dumps(input_images_dict),
                           weights_paths=json.dumps(weights_results),
                           grad_cam_paths=json.dumps(grad_cam_results),
                           saliency_map_paths=json.dumps(saliency_map_results))


def inspector(model, images, number_of_classes, engine="keras", grad_cam_classes=None):
    global prediction_model
    global nodes
    global edges

    if engine == "keras":
        from tensorflow import keras
        if not isinstance(model, keras.models.Sequential) or not isinstance(model, keras.models.Model):
            return "Engine is set to KERAS but detected model is not Keras. Please check."
    elif engine == "pytorch":
        import torch.nn as nn
        if not isinstance(model, nn.Module):
            return "Engine is set to PYTORCH but detected model is not PyTorch. Please check."
    else:
        return "Engine can be only: keras or pytorch"

    if engine == "keras":
        outputs = [layer.output for layer in model.layers]
        # Graph builder
        from cnn_raccoon.graph.graph_builder import keras_graph
        keras_graph(model)

    elif engine == "pytorch":

        # Check for input images
        import torch
        import torch.nn as nn

        if len(images) > 0:
            if not isinstance(images[0], torch.Tensor):
                return None
        else:
            return None

        # Process input images
        from cnn_raccoon.pytorch.utils import save_input_images
        save_input_images(images)

        from cnn_raccoon.pytorch.utils import module2traced
        if len(images[0].shape) == 3:
            img = images[0].unsqueeze(0)
        else:
            img = images[0]

        # Trace model from nested PyTorch to linear layer order
        traced_model = module2traced(model, img)

        # Graph builder
        from cnn_raccoon.graph.graph_builder import pytorch_graph
        pytorch_graph(traced_model)

        from cnn_raccoon.pytorch.model_weights import Weights
        weights = Weights(model)

        print("[INFO: ] Weights calculation based on input images has started.")
        from cnn_raccoon.pytorch.calculations import weights_inspector
        weights_inspector(weights, images)

        from cnn_raccoon.pytorch.grad_cam import GradCam
        grad_cam = GradCam(model)

        if grad_cam_classes is None:
            if number_of_classes > 10:
                print("[INFO: ] For memory reason, max number of classes is 10. "
                      "Library will randomly sample 10 classes for you.")
                grad_cam_classes = np.random.choice(number_of_classes, size=10, replace=False)
            else:
                grad_cam_classes = np.arange(0, number_of_classes)

        print("[INFO: ] Grad Cam calculation based on input images has started.")
        from cnn_raccoon.pytorch.calculations import grad_cam_inspector
        grad_cam_inspector(grad_cam, images, grad_cam_classes)

        from cnn_raccoon.pytorch.saliency_map import SaliencyMap
        model.eval()
        saliency_map = SaliencyMap(model)

        print("[INFO: ] Saliency Map calculation based on input images has started.")
        from cnn_raccoon.pytorch.calculations import saliency_map_inspector
        saliency_map_inspector(saliency_map, images)

    app.run(port=5000)
