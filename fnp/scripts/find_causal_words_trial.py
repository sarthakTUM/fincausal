from pathlib import Path

from fnp.fincausal.data_types.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter
from fnp.fincausal.data_types.model import FinCausalTask1FindCausalWords

# 1. specify a file to read
from fnp.fincausal.evaluation.metrics import ConfMatrixCalculator, Metrics

fincausal_task1_train_file_path = Path('/media/sarthak/HDD/data_science/fnp/resources/fnp2020-fincausal-task1-train.csv')
fincausal_task1_val_file_path = Path('/media/sarthak/HDD/data_science/fnp/resources/fnp2020-fincausal-task1-val.csv')
fincausal_task1_test_file_path = Path('/media/sarthak/HDD/data_science/fnp/resources/fnp2020-fincausal-task1-test.csv')

# 2. load the file using FileReader
fincausal_task1_train_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_train_file_path)
fincausal_task1_val_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_val_file_path)
fincausal_task1_test_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_test_file_path)

fincausal_task1_train_dataset = fincausal_task1_train_dataset_csv_reader.read()
fincausal_task1_val_dataset = fincausal_task1_val_dataset_csv_reader.read()
fincausal_task1_test_dataset = fincausal_task1_test_dataset_csv_reader.read()


# 3. specify the list of features to be extracted
feature_extractors = []

# 4. create modeling instances from the dataset_instances and the feature_extractors
fincausal_task1_train_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_train_dataset,
                                                                                     feature_extractors=feature_extractors)
fincausal_task1_val_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_val_dataset,
                                                                                   feature_extractors=feature_extractors)
fincausal_task1_test_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_test_dataset,
                                                                                    feature_extractors=feature_extractors)

fincausal_task1_train_modeling_dataset = fincausal_task1_train_modeling_dataset_reader.read()
fincausal_task1_val_modeling_dataset = fincausal_task1_val_modeling_dataset_reader.read()
fincausal_task1_test_modeling_dataset = fincausal_task1_test_modeling_dataset_reader.read()


# 5. Create a Model
find_causal_words_model = FinCausalTask1FindCausalWords()

# 6. fit the model to the data
find_causal_words_model.predict_on_dataset(dataset=fincausal_task1_test_modeling_dataset)

# 7 evaluate the predictions
tp, fp, fn = ConfMatrixCalculator.calculate_tp_fp_fn_task1(dataset=fincausal_task1_test_modeling_dataset)
print(tp, fp, fn)
f1 = Metrics.f_beta(tp, fp, fn, beta=1.0)
recall = Metrics.recall(tp, fn)
print(f1, recall)

