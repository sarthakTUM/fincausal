from pathlib import Path
from typing import List, Optional
from fnp.fincausal.data_types.core import FinCausalDataset
from fnp.fincausal.data_types.dataset_instance import FinCausalTask1DatasetInstance, FinCausalTask2DatasetInstance
from fnp.fincausal.data_types.modeling_instance import FinCausalTask1ModelingInstance


class FinCausalTask1Dataset(FinCausalDataset):

    def __init__(self, instances: List[FinCausalTask1DatasetInstance],
                 path: Optional[Path] = None):
        """
        :param path: path to the CSV file containing the raw dataset for the task 1 of FinCausal task
        """
        super().__init__(instances=instances)
        self.path = path


class FinCausalTask2Dataset(FinCausalDataset):
    """
    This class is not used because we are not working on Task 2
    """

    def __init__(self, instances: List[FinCausalTask2DatasetInstance]):
        super().__init__(instances=instances)


class FinCausalTask1ModelingDataset(FinCausalDataset):

    def __init__(self, instances: List[FinCausalTask1ModelingInstance]):
        super(FinCausalTask1ModelingDataset, self).__init__(instances=instances)

    @property
    def is_labeled(self) -> bool:
        return all([True if modeling_instance.gold is not None else False for modeling_instance in self.instances])

