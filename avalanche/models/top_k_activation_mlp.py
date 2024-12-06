################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Alberto Confente                                                  #                                          #
# Website: avalanche.continualai.org                                           #
################################################################################

import torch.nn as nn
import torch

from avalanche.models.base_model import BaseModel


class TopKActivationMLP(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.

    **Example**::

        >>> from avalanche.models import SimpleMLP
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleMLP(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=2,
        drop_rate=0.0,
        k=256
    ):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size),
                TopKActivation(k),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size),
                        TopKActivation(k),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x
    
class TopKActivation(nn.Module):
    def __init__(self, k):
        """
        Initializes the TopKActivation module.
        Args:
            k (int): The number of largest values to keep.
        """
        super(TopKActivation, self).__init__()
        self.k = k

    def forward(self, x):
        """
        Forward pass of the TopKActivation module.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor with the largest k values retained.
        """
        flat_x = x.view(x.size(0), -1)
        _, topk_indices = torch.topk(flat_x, self.k, dim=1)
        mask = torch.zeros_like(flat_x).scatter(1, topk_indices, 1)
        mask = mask.view_as(x)
        return x * mask


__all__ = ["kWtaMLP"]
