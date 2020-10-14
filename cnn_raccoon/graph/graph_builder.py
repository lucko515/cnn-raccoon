"""
This file contains JavaScript Graph builders for Keras (TensorFlow) and PyTorch models.

NOTE: Dependencies are imported in functions in purpose so people don't have to
      install both TensorFlow and PyTorch on their machines.
"""

from cnn_raccoon import nodes, edges, layers_dict, layer_info_dict


def keras_graph(model):
    from tensorflow import keras

    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            nodes.append({"data": {"id": layer.name, "name": layer.name, "faveColor": '#F5A45D',
                                   "faveShape": 'rectangle'}})
        elif isinstance(layer, keras.layers.Dense):
            nodes.append({"data": {"id": layer.name, "name": layer.name, "faveColor": '#F5A45D',
                                   "faveShape": 'ellipse'}})
        elif isinstance(layer, keras.layers.MaxPool2D):
            nodes.append({"data": {"id": layer.name, "name": layer.name, "faveColor": '#F5A45D',
                                   "faveShape": 'diamond'}})
        else:
            nodes.append({"data": {"id": layer.name, "name": layer.name, "faveColor": '#F5A45D',
                                   "faveShape": 'hexagon'}})

        layers_dict[layer.name] = layer
        layer_info_dict[layer.name] = str(layer.get_config())

    for n in range(len(nodes) - 1):
        node = nodes[n]
        next_node = nodes[n + 1]
        edges.append(
            {"data": {"source": node["data"]["id"], "target": next_node["data"]["id"], "faveColor": '#6FB1FC'}})


def pytorch_graph(layers):
    import torch.nn as nn
    # Graph builder
    for l in range(len(layers)):
        layer = layers[l]
        name = "layer_{}_".format(l)
        if isinstance(layer, nn.Conv2d):
            name += layer.__class__.__name__
            nodes.append({"data": {"id": name, "name": name, "faveColor": '#F5A45D',
                                   "faveShape": 'rectangle', "isConv": "true"}})
        elif isinstance(layer, nn.Linear):
            name += layer.__class__.__name__
            nodes.append({"data": {"id": name, "name": name, "faveColor": '#F5A45D',
                                   "faveShape": 'ellipse'}})
        elif isinstance(layer, nn.MaxPool2d):
            name += layer.__class__.__name__
            nodes.append({"data": {"id": name, "name": name, "faveColor": '#F5A45D',
                                   "faveShape": 'diamond'}})
        else:
            name += layer.__class__.__name__
            nodes.append({"data": {"id": name, "name": name, "faveColor": '#F5A45D',
                                   "faveShape": 'hexagon'}})

        layers_dict[name] = layers[l]
        layer_info_dict[name] = str(layers[l])

    for n in range(len(nodes) - 1):
        node = nodes[n]
        next_node = nodes[n + 1]
        edges.append(
            {"data": {"source": node["data"]["id"], "target": next_node["data"]["id"], "faveColor": '#6FB1FC'}})
