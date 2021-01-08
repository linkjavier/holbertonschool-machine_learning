#!/usr/bin/env python3
""" Tensorflow module """

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Forward propagation function """
    for LayerSize, activation in zip(layer_sizes, activations):
        x = create_layer(x, LayerSize, activation)
    return x
