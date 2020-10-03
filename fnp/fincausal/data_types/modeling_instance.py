from typing import List, Optional
from fnp.fincausal.data_types.core import Feature
from fnp.fincausal.data_types.core import FinCausalModelingInstance
from fnp.fincausal.data_types.dataset_instance import FinCausalTask1DatasetInstance
from fnp.fincausal.data_types.label import FinCausalTask1Label, FinCausalTask1Pred


class FinCausalTask1ModelingInstance(FinCausalModelingInstance):
    def __init__(self,
                 dataset_instance: FinCausalTask1DatasetInstance,
                 features: List[Feature],
                 label: FinCausalTask1Label,
                 pred: Optional[FinCausalTask1Pred] = None):
        super(FinCausalTask1ModelingInstance, self).__init__(dataset_instance=dataset_instance, features=features)
        self.label = label
        self.pred = pred
