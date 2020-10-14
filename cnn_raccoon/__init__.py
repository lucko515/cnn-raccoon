import os
import json
import shutil

import numpy as np

lib_path = os.path.dirname(os.path.abspath(__file__))
images_top_dir = lib_path + "/static/images/project"
img_relative_path = "images/project"

# Third party libraries
from flask import Flask, render_template

# Create Flask application
app = Flask(__name__)

# Global variables
prediction_model = None
backend = None
nodes = []  # List for Nodes in the CNN graph
edges = []  # Edges in the CNN graph
layers_dict = {}  # Python dict containing all layers
layer_info_dict = {}  # Layer information (hyperparams)
weights_results = {}   # Map between weights images stored locally and paths
grad_cam_results = {}  # Map between grad cam images stored locally and paths
saliency_map_results = {}  # Map between saliency maps images stored locally and paths
input_images_dict = {}  # Map between input images images stored locally and paths

# Make sure to create image-project folder for the current session
if os.path.exists(images_top_dir):
    shutil.rmtree(images_top_dir, ignore_errors=True)
    os.mkdir(images_top_dir)
else:
    os.mkdir(images_top_dir)


# Home/login screen
@app.route("/", methods=['GET'])
def index():
    """
    Index page for Dashboard rendering.

    :return: rendered dashboard template.
    """
    return render_template("dashboard.html",
                           nodes=json.dumps(nodes), edges=json.dumps(edges),
                           backend=backend,
                           layer_info_dict=layer_info_dict,
                           input_images=json.dumps(input_images_dict),
                           weights_paths=json.dumps(weights_results),
                           grad_cam_paths=json.dumps(grad_cam_results),
                           saliency_map_paths=json.dumps(saliency_map_results))


def inspector(model,
              images,
              number_of_classes,
              engine="keras",
              grad_cam_classes=None,
              port=5000):
    """
    Inspector will analyze your trained CNN and provide insights in easy to use dashboard.


    :param model: Object of a trained model (Pytorch or Keras)
    :param images: Images used for model analysis. NOTE: Input depends on the Engine selected!
                    Case 1 - Keras: Load images in numpy array.
                                    Shape example: Grayscale: [5, 32, 32, 1] or RGB: [5, 32, 32, 3]
                    Case 2 - Pytorch: Load images in format of PyTorch tensors. The same way you would before training.
                                    Cifar10 example:
                                    transform = transforms.Compose(
                                                        [ transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                                    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                                            download=True, transform=transform)
                                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                                              shuffle=True, num_workers=2)

    :param number_of_classes: Integer, number of classes in a dataset.
                                Example: Cifar10 - 10, Dogs Vs Cats - 2
    :param engine: "pytorch" or "keras"
    :param grad_cam_classes: List of classes that will be used to run GradCam algorithm. Max=10
                            NOTE: If your list is longer then 10 it will truncate it to 10 (randomly)
    :param port: Dashboard port. Default=5000
    """
    global prediction_model
    global nodes
    global edges
    global backend

    backend = engine
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

    if grad_cam_classes is None:
        if number_of_classes > 10:
            print("[INFO: ] For memory reason, max number of classes is 10. "
                  "Library will randomly sample 10 classes for you.")
            grad_cam_classes = np.random.choice(number_of_classes, size=10, replace=False)
        else:
            grad_cam_classes = np.arange(0, number_of_classes)

    if engine == "keras":
        from tensorflow import keras
        from tensorflow.keras.models import Model, Sequential

        print(type(images))
        if len(images) > 0:
            if not isinstance(images[0], np.ndarray):
                return None
        else:
            return None

        # Process input images
        from cnn_raccoon.keras.utils import save_input_images
        save_input_images(images)

        # Create weights-model
        from cnn_raccoon.keras.utils import create_weights_model
        output_model, output_names = create_weights_model(model)

        # Graph builder
        print("[INFO: ] Model's graph created.")
        from cnn_raccoon.graph.graph_builder import keras_graph
        keras_graph(model)

        print("[INFO: ] Weights calculation based on input images has started.")
        from cnn_raccoon.keras.calculations import weights_inspector
        weights_inspector(output_model, images, output_names)

        print("[INFO: ] Grad Cam calculation based on input images has started.")
        from cnn_raccoon.keras.calculations import grad_cam_inspector
        grad_cam_inspector(model, images, grad_cam_classes)

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
        print("[INFO: ] Model's graph created.")
        from cnn_raccoon.graph.graph_builder import pytorch_graph
        pytorch_graph(traced_model)

        from cnn_raccoon.pytorch.model_weights import Weights
        weights = Weights(model)

        print("[INFO: ] Weights calculation based on input images has started.")
        from cnn_raccoon.pytorch.calculations import weights_inspector
        weights_inspector(weights, images)

        from cnn_raccoon.pytorch.grad_cam import GradCam
        grad_cam = GradCam(model)

        print("[INFO: ] Grad Cam calculation based on input images has started.")
        from cnn_raccoon.pytorch.calculations import grad_cam_inspector
        grad_cam_inspector(grad_cam, images, grad_cam_classes)

        from cnn_raccoon.pytorch.saliency_map import SaliencyMap
        model.eval()
        saliency_map = SaliencyMap(model)

        print("[INFO: ] Saliency Map calculation based on input images has started.")
        from cnn_raccoon.pytorch.calculations import saliency_map_inspector
        saliency_map_inspector(saliency_map, images)

    app.run(port=port)
