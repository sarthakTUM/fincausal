from pathlib import Path

from fnp.fincausal.data_types.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter
from fnp.fincausal.data_types.feature_extractors import IsCausalConnectivePresentFeatureExtractor
from fnp.fincausal.data_types.model import FinCausalTask1SklearnLogRegModel

# 1. specify a file to read
fincausal_task1_file_path = Path('/media/sarthak/HDD/data_science/fnp/resources/fnp2020-fincausal-task1.csv')

# 2. load the file using FileReader
fincausal_task1_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_file_path)
fincausal_task1_dataset = fincausal_task1_dataset_csv_reader.read()


# 3. specify the list of features to be extracted
feature_extractors = [IsCausalConnectivePresentFeatureExtractor()]

# 4. create modeling instances from the dataset_instances and the feature_extractors
fincausal_task1_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_dataset,
                                                                               feature_extractors=feature_extractors)
fincausal_task1_modeling_dataset = fincausal_task1_modeling_dataset_reader.read()


# 5. Create a Model
sklearn_logreg_model = FinCausalTask1SklearnLogRegModel()

# 6. fit the model to the data
sklearn_logreg_model.fit(fincausal_task1_modeling_dataset)

