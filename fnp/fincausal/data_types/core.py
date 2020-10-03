from abc import ABC, abstractmethod
from typing import Optional, List, Union
import random


class Feature:
    """
    Base Class for the features
    """
    def __init__(self):
        pass


class FinCausalLabel:
    """
    Base class for labels for the FinCausal Task
    """
    def __init__(self):
        pass


class FinCausalDatasetInstance:
    """
    Base class for an instance of the FinCausal task
    """
    def __init__(self,
                 unique_id: str,
                 index: str,
                 text: str
                 )-> None:
        """
        inits an instance
        :param unique_id: custom generated unique_id. This is different from the unique_id (index) provided by the
        FinCausal.
        :param index: Index (ID) provided by FinCausal
        :param text: text segment
        """
        self.unique_id = unique_id
        self.index: str = index
        self.text: str = text


class FinCausalModelingInstance:
    """
    Base class for modeling instance that contains the features (rule based) extracted from the FinCausalDatasetInstance
    """

    def __init__(self,
                 dataset_instance: FinCausalDatasetInstance,
                 features: Optional[List[Feature]] = None):
        """
        inits a modeling instance
        :param dataset_instance: FinCausalDatasetInstance
        :param features: features of type `Feature` extracted by the `FeatureExtractor` from the FinCausalDatasetInstance
        """
        self.dataset_instance = dataset_instance
        self.features = features


class FinCausalDataset:
    """
    Base Class for FinCausal Dataset containing the list of FinCausalDatasetInstance or FinCausalModelingInstance
    """
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
        """
        sampling instances from the dataset randomly
        :param n: number of samples to randomly sample
        :return: subset FinCausalDataset
        """
        return FinCausalDataset(instances=random.sample(self.instances, n))


class FeatureExtractor(ABC):
    """
    Base Class for extracting the features from a FinCausalDatasetInstance
    """
    @abstractmethod
    def extract(self, dataset_instance: FinCausalDatasetInstance) -> Feature:
        """
        override this function to run on the dataset_instance and extract a `Feature` form it
        :param dataset_instance:
        :return:
        """
        raise NotImplementedError


class Model(ABC):
    """
    Base Class for the models. Follows the SKlearn API.
    """

    @abstractmethod
    def fit(self, dataset: FinCausalDataset):
        """
        fits model on dataset without evaluating (when validation set is not available)
        """
        raise NotImplementedError

    @abstractmethod
    def fit_and_evaluate(self,
                         train_dataset: FinCausalDataset,
                         val_dataset: FinCausalDataset):
        """
        fits model on dataset with evaluating (when validation set is available)
        """
        raise NotImplementedError


class DatasetAdapter(ABC):

    """
    Base Class for adapting 1 dataset type to another
    """
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def read(self) -> FinCausalDataset:
        raise NotImplementedError

