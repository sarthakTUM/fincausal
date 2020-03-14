from pathlib import Path

from fnp.fincausal.data_types.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter
from fnp.fincausal.data_types.model import SimpleTransformersModel

from fnp.fincausal.evaluation.metrics import ConfMatrixCalculator, Metrics


# 1. specify a file to read
fincausal_task1_predict_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/iteration_1/test.csv')

# 2. load the file using FileReader
fincausal_task1_predict_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_predict_file_path)
fincausal_task1_predict_dataset = fincausal_task1_predict_dataset_csv_reader.read()
print('test: {}'.format(len(fincausal_task1_predict_dataset)))


# 3. specify the list of features to be extracted
feature_extractors = []

# 4. create modeling instances from the dataset_instances and the feature_extractors
fincausal_task1_predict_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_predict_dataset,
                                                                                       feature_extractors=feature_extractors)

fincausal_task1_predict_modeling_dataset = fincausal_task1_predict_modeling_dataset_reader.read()


# 5. Predict on the test dataset
simpletransformers_model = SimpleTransformersModel(
    model_type='distilbert',
    model_name_or_path=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/6/iteration_1/output/best_model'),
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/6/iteration_1/output/best_model'))

simpletransformers_model.predict_on_dataset(dataset=fincausal_task1_predict_modeling_dataset)

# 7 evaluate the predictions
tp, fp, fn, tn = ConfMatrixCalculator.calculate_tp_fp_fn_task1(dataset=fincausal_task1_predict_modeling_dataset)
print(tp, fp, fn, tn)
f1 = Metrics.f_beta(tp, fp, fn, beta=1.0)
recall = Metrics.recall(tp, fn)
precision = Metrics.precision(tp, fp)
print('f1: {} recall: {} precision: {}'.format(f1, recall, precision))
