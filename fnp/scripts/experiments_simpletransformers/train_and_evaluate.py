from pathlib import Path

from fnp.fincausal.data_types.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter
from fnp.fincausal.data_types.model import SimpleTransformersModel

from fnp.fincausal.evaluation.metrics import ConfMatrixCalculator, Metrics
from fnp.fincausal import utils

# 1. specify a file to read
fincausal_task1_train_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/iteration_1/train_with_semeval.csv')
fincausal_task1_val_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/iteration_1/val.csv')
fincausal_task1_test_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/iteration_1/test.csv')

# 2. load the file using FileReader
fincausal_task1_train_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_train_file_path)
fincausal_task1_val_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_val_file_path)
fincausal_task1_test_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_test_file_path)

fincausal_task1_train_dataset = fincausal_task1_train_dataset_csv_reader.read()
fincausal_task1_val_dataset = fincausal_task1_val_dataset_csv_reader.read()
fincausal_task1_test_dataset = fincausal_task1_test_dataset_csv_reader.read()
print('train: {}, val: {}, test: {}'.format(len(fincausal_task1_train_dataset),
                                            len(fincausal_task1_val_dataset),
                                            len(fincausal_task1_test_dataset)))


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


# 5. Create a Model (check names for HF models here: https://huggingface.co/transformers/pretrained_models.html
simpletransformers_model = SimpleTransformersModel(
    model_type='distilbert',
    model_name_or_path='distilbert-base-cased',
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/8/iteration_1'),
    class_weights=utils.calculate_class_weights_task1(fincausal_task1_train_modeling_dataset))

# 6. fit the model to the data
simpletransformers_model.fit_and_evaluate(train_dataset=fincausal_task1_train_modeling_dataset,
                                          val_dataset=fincausal_task1_val_modeling_dataset,
                                          num_train_epochs=15)

# 8. Predict on the test dataset
simpletransformers_model = SimpleTransformersModel(
    model_type='distilbert',
    model_name_or_path=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/8/iteration_1/output/best_model'),
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/8/iteration_1/output/best_model'))

simpletransformers_model.predict_on_dataset(dataset=fincausal_task1_test_modeling_dataset)

# 7 evaluate the predictions
tp, fp, fn, tn = ConfMatrixCalculator.calculate_tp_fp_fn_task1(dataset=fincausal_task1_test_modeling_dataset)
print(tp, fp, fn, tn)
f1 = Metrics.f_beta(tp, fp, fn, beta=1.0)
recall = Metrics.recall(tp, fn)
precision = Metrics.precision(tp, fp)
print('f1: {} recall: {} precision: {}'.format(f1, recall, precision))

