from typing import Union, List, Tuple
from torch import Tensor
from torch_geometric.data import Dataset

import torch
import numpy as np

from torch_geometric.io.planetoid import index_to_mask


def get_idx_split(
        dataset: Dataset,
        samples_per_class_in_one_fold: Union[int, float] = None,
        k_fold: int = None,
        test_samples_per_class: Union[int, float] = None) -> Tuple[List[np.array], np.array]:
    """utility function for creating k-fold cross-validation split for a dataset.
    The split is either created by specifying the number or fraction of samples per class in one fold. 
    If the fraction of samples per class is chosen, the fraction is relative to the number of labeled 
    data points for each class separately. 
    Within the k-fold cross-validation set one fold is to used as validation set. The rest bigger 
    portions are used as training set.
    Part of the code taken from (https://github.com/shchur/gnn-benchmark)
    Args:
        dataset: Dataset
        samples_per_class_in_one_fold: Union[int, float], number of fraction of samples per class in one fold.
        k_fold: int, number of folds for internal cross-validation which follows the paper 
            "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration"
        test_samples_per_class: Union[int, float], number or fraction of samples per cleass in the test set.
    Returns:
        k_fold_indices: list of k one-fold indices
        test_indices: indicies of test set
    """

    data = dataset.data

    assert samples_per_class_in_one_fold is not None
    assert k_fold is not None
    assert test_samples_per_class is not None

    labels = data.y
    num_nodes = labels.size(0)
    num_classes = max(labels) + 1
    classes = range(num_classes)
    remaining_indices = list(range(num_nodes))
    k_fold_indices = []

    classes = [c for c in classes]

    # k-fold indices for training and validation set
    fold_indices = sample_per_class(labels, num_nodes, classes,
                                    samples_per_class_in_one_fold*k_fold, forbidden_indices=None)
    for k in range(k_fold):
        k_fold_indices.append(fold_indices[k::k_fold])
        # print((labels[fold_indices[k::k_fold]] == 2).sum())
    forbidden_indices = np.concatenate((forbidden_indices, fold_indices))

    # test indices (exclude test indices)
    if test_samples_per_class is not None:
        test_indices = sample_per_class(labels, num_nodes, classes,
                                        test_samples_per_class,
                                        forbidden_indices=forbidden_indices)
    else:
    # All the remaining indices belong to test indices
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    for i in range(k_fold):
        assert len(set(k_fold_indices[i])) == len(k_fold_indices[i])
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    for i in range(k_fold):
        for j in range(i+1, k_fold):
            assert len(set(k_fold_indices[i]) - set(k_fold_indices[j])) == len(set(k_fold_indices[i]))
            assert len(set(k_fold_indices[i]) - set(test_indices)) == len(set(k_fold_indices[i]))

    return k_fold_indices, test_indices

 
def sample_per_class(labels: Tensor, num_nodes: int, classes: List[int],
                     samples_per_class: Union[int, float],
                     forbidden_indices: np.array = None) -> np.array:
    """samples a subset of indices based on specified number of samples per class
    Args:
        labels (Tensor): tensor of ground-truth labels
        num_nodes (int): number nof nodes
        classes (List[int]): classes (labels) for which the subset is sampled
        samples_per_class (Union[int, float]): number or fraction of samples per class
        forbidden_indices (np.array, optional): indices to ignore for sampling. Defaults to None.
    Returns:
        np.array: sampled indices
    """

    sample_indices_per_class = {index: [] for index in classes}
    num_samples_per_class = {index: None for index in classes}

    # get indices sorted by class
    for class_index in classes:
        for sample_index in range(num_nodes):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    for class_index in classes:
        if isinstance(samples_per_class, float):
            class_labels = sample_indices_per_class[class_index]
            num_samples_per_class[class_index] = int(samples_per_class * len(class_labels))
        else:
            num_samples_per_class[class_index] = samples_per_class

    # get specified number of indices for each class
    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_samples_per_class[class_index], replace=False)
         for class_index in classes
        ])