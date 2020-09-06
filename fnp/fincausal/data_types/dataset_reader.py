from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from pathlib import Path

from fnp.fincausal.data_types.core import FinCausalDataset, FeatureExtractor
from fnp.fincausal.data_types.dataset import FinCausalTask1Dataset, FinCausalTask1ModelingDataset
from fnp.fincausal.data_types.dataset_instance import FinCausalTask1DatasetInstance
from fnp.fincausal.data_types.label import FinCausalTask1Label
from fnp.fincausal.data_types.modeling_instance import FinCausalTask1ModelingInstance

import tqdm


class DatasetAdapter(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def read(self) -> FinCausalDataset:
        raise NotImplementedError


class CSVAdapter(DatasetAdapter):

    def __init__(self, csv_filepath: Path):
        super(CSVAdapter, self).__init__()
        self.csv_filepath = csv_filepath

    @abstractmethod
    def read(self) -> FinCausalDataset:
        raise NotImplementedError


class FinCausalTask1DatasetCSVAdapter(CSVAdapter):

    def __init__(self,
                 csv_filepath: Path) -> None:
        super().__init__(csv_filepath=csv_filepath)

    def read(self) -> FinCausalTask1Dataset:
        """
        reads the FinCausalTask1 dataset
        :return: Dataset with a list of dataset instances.
        """
        df = pd.read_csv(self.csv_filepath, sep=',')
        with_label = 'Gold' in df.columns
        list_of_instances = []
        for index, row in df.iterrows():
            if with_label:
                instance = FinCausalTask1DatasetInstance(
                    unique_id=row['unique_id'],
                    index=row['Index'],
                    text=row['Text'],
                    gold=row['Gold'])
            else:
                instance = FinCausalTask1DatasetInstance(
                    unique_id=row['unique_id'],
                    index=row['Index'],
                    text=row['Text'],
                    gold=None)
            list_of_instances.append(instance)
        return FinCausalTask1Dataset(list_of_instances,
                                     path=self.csv_filepath)


class FinCausalTask1ModelingDatasetAdapter(DatasetAdapter):

    def __init__(self,
                 fincausal_task1_dataset: FinCausalTask1Dataset,
                 feature_extractors: List[FeatureExtractor]):
        super(FinCausalTask1ModelingDatasetAdapter, self).__init__()
        self.fincausal_task1_dataset = fincausal_task1_dataset
        self.feature_extractors = feature_extractors

    def read(self) -> FinCausalTask1ModelingDataset:
        list_of_modeling_instances = []
        for dataset_instance in tqdm.tqdm(self.fincausal_task1_dataset.instances):
            features_for_dataset_instance = []
            for feature_extractor in self.feature_extractors:
                features_for_dataset_instance.append(feature_extractor.extract(dataset_instance=dataset_instance))
            list_of_modeling_instances.append(FinCausalTask1ModelingInstance(dataset_instance=dataset_instance,
                                                                             features=features_for_dataset_instance,
                                                                             label=FinCausalTask1Label(
                                                                                 causal=dataset_instance.gold)))
        return FinCausalTask1ModelingDataset(instances=list_of_modeling_instances)
