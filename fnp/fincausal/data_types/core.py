from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict
import random


class Feature:
    def __init__(self):
        pass


class FinCausalLabel:
    def __init__(self):
        pass


class FinCausalDatasetInstance:
    def __init__(self,
                 unique_id: str,
                 index: str,
                 text: str
                 )-> None:
        self.unique_id = unique_id
        self.index: str = index
        self.text: str = text


class FinCausalModelingInstance:

    def __init__(self,
                 dataset_instance: FinCausalDatasetInstance,
                 features: Optional[List[Feature]] = None):
        self.dataset_instance = dataset_instance
        self.features = features


class FinCausalDataset:
    def __init__(self, instances: Union[List[FinCausalDatasetInstance], List[FinCausalModelingInstance]]):
        self.instances = instances
        self.size = len(instances)

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def __len__(self):
        return self.size

    def sample(self, n: int):
        return FinCausalDataset(instances=random.sample(self.instances, n))


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, dataset_instance: FinCausalDatasetInstance) -> Feature:
        raise NotImplementedError


class Model(ABC):

    @abstractmethod
    def fit(self, dataset: FinCausalDataset):
        raise NotImplementedError

    @abstractmethod
    def fit_and_evaluate(self,
                         train_dataset: FinCausalDataset,
                         val_dataset: FinCausalDataset):
        raise NotImplementedError

