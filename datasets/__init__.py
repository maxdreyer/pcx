from typing import Callable

from datasets.imagenet_pexels import get_imagenet_pexels

DATASETS = {
    "imagenet_pexels": get_imagenet_pexels,
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        print(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")