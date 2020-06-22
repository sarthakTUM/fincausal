from pathlib import Path
from typing import List, Optional
from fnp.fincausal.data_types.core import FinCausalDataset
from fnp.fincausal.data_types.dataset_instance import FinCausalTask1DatasetInstance, FinCausalTask2DatasetInstance
from fnp.fincausal.data_types.modeling_instance import FinCausalTask1ModelingInstance


class FinCausalTask1Dataset(FinCausalDataset):

    def __init__(self, instances: List[FinCausalTask1DatasetInstance],
                 path: Optional[Path] = None):
        super().__init__(instances=instances)
        self.path = path


class FinCausalTask2Dataset(FinCausalDataset):

    def __init__(self, instances: List[FinCausalTask2DatasetInstance]):
        super().__init__(instances=instances)


class FinCausalTask1ModelingDataset(FinCausalDataset):

    def __init__(self, instances: List[FinCausalTask1ModelingInstance]):
        super(FinCausalTask1ModelingDataset, self).__init__(instances=instances)

