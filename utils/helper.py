import os

import torch
import yaml
from crp.helper import get_layer_names
from typing import List


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}
        config["wandb_id"] = os.path.basename(config_path)[:-5]
    return config


def get_layer_names_model(model: torch.nn.Module, model_name: str) -> List[str]:
    """
    Get layer names of a model.
    :param model:   model
    :param model_name:  model name (e.g. vgg16)
    :return:
    """
    if "vgg" in model_name:
        layer_names = get_layer_names(model, [torch.nn.Conv2d,
                                              # torch.nn.Linear
                                              ])
    elif "resnet" in model_name:
        layer_names = get_layer_names(model, [InspectionLayer])
    elif "efficientnet" in model_name:
        layer_names = get_layer_names(model, [torch.nn.Identity])
    elif "vit" in model_name:
        layer_names = get_layer_names(model, [torch.nn.Identity])
    else:
        raise NotImplementedError
    return layer_names


class InspectionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
