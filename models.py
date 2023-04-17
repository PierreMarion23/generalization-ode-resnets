import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

import utils


def create_linear_layers_rbf(
        depth: int, width: int, bandwidth: float) -> nn.Sequential:
    """Initialize the weights of a sequence of layers of fixed width as
    discretizations of a smooth Gaussian process with rbf kernel.

    :param depth: depth of the ResNet
    :param width: width of the layers
    :param regularity: variance of the rbf kernel
    :return: initialized layers of the ResNet as a nn.Sequential object
    """
    mean = [0] * (depth + 1)
    cov_matrix = utils.cov_matrix_for_rbf_kernel(depth, bandwidth)
    weights = np.random.default_rng().multivariate_normal(
        mean, cov_matrix, (width, width)) / np.sqrt(width)
    layers = [
        nn.Linear(width, width, bias=False) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(torch.Tensor(weights[:, :, k]))
    return nn.Sequential(*layers)



def create_linear_layer(
        in_features: int, out_features:int, bias: bool = True) -> nn.Linear:
    """Initialize one linear layer with a normal distribution of
    variance 1 / in_features

    :param in_features: size of the input to the layer
    :param out_features: size of the output of the layer
    :param bias: whether to include a bias
    :return:
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    layer.weight = nn.Parameter(
        torch.randn(out_features, in_features) / np.sqrt(in_features))
    if bias:
        layer.bias = nn.Parameter(
            torch.randn(out_features,) / np.sqrt(in_features)) 
    return layer


class ResNet(pl.LightningModule):
    def __init__(self, first_coord: int, final_width:int,
                 **model_config: dict):
        """General class of residual neural network

        :param first_coord: size of the input data
        :param final_width: size of the output
        :param model_config: configuration dictionary with hyperparameters
        """
        super().__init__()

        self.initial_width = first_coord
        self.final_width = final_width
        self.model_config = model_config
        self.width = model_config['width']
        self.depth = model_config['depth']
        self.activation = getattr(nn, model_config['activation'])()

        # Uniform initialization on [-sqrt(3/width), sqrt(3/width)]
        self.init = create_linear_layer(
            self.initial_width, self.width, bias=False)
        self.final = create_linear_layer(
            self.width, self.final_width, bias=False)
        self.outer_weights = create_linear_layers_rbf(
                self.depth, self.width, model_config['rbf_bandwidth'])

        self.loss = nn.CrossEntropyLoss()

    def forward_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Function that outputs the last hidden state, useful to compare norms

        :param hidden_state: output of the initial layer
        :return: output of the last hidden layer
        """
        for k in range(self.depth):
            hidden_state = hidden_state + \
                           self.outer_weights[k](self.activation(hidden_state)) / float(self.depth)
        return hidden_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.init(x)
        hidden_state = self.forward_hidden_state(hidden_state)
        return self.final(hidden_state)

    def training_step(self, batch, batch_no):
        self.train()
        data, target = batch
        logits = self(data)
        loss = self.loss(logits, target)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.model_config['lr'])
        return {"optimizer": optimizer}
