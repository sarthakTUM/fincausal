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

