from typing import List

from sklearn.utils import class_weight
import numpy as np

from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset


def calculate_class_weights_task1(modeling_dataset: FinCausalTask1ModelingDataset) -> List[float]:

    labels = []
    for modeling_instance in modeling_dataset:
        labels.append(modeling_instance.label.causal)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(labels),
                                                      labels)
    return list(class_weights)


def softmax(x: List[float]) -> List[float]:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
