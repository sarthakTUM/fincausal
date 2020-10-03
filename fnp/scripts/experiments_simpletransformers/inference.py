from pathlib import Path
from fnp.fincausal.evaluation.inference import FinCausalTask1Inference
from fnp.fincausal.evaluation.metrics import MetricsWrapper
from fnp.fincausal.model.simpletransformers_model import SimpleTransformersModel
from fnp.fincausal.preprocessing.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter

# 0. Constants
evaluate = True

# 1. specify a file to read
fincausal_task1_predict_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/test.csv')

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
    model_type='bert',
    model_name_or_path=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/145_1/checkpoint-1116-epoch-1'),
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/145_1/checkpoint-1116-epoch-1'))

simpletransformers_model.predict_on_dataset(dataset=fincausal_task1_predict_modeling_dataset)

# 8. evaluate the predictions
task1_metrics = None
task1_cm = None
if evaluate:
    task1_metrics = MetricsWrapper.calculate_metrics_task1(dataset=fincausal_task1_predict_modeling_dataset)
    task1_cm = MetricsWrapper.calculate_confusionmatrix_task1(dataset=fincausal_task1_predict_modeling_dataset)
    print(task1_metrics)
    print(task1_cm)

fincausal_task1_inference = FinCausalTask1Inference(
    model_path=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/145_1/checkpoint-1116-epoch-1'),
    predict_file_path=fincausal_task1_predict_file_path,
    f1=task1_metrics['f1'] if task1_metrics else None,
    recall=task1_metrics['recall'] if task1_metrics else None,
    precision=task1_metrics['precision'] if task1_metrics else None,
    tp=task1_cm['tp'] if task1_cm else None,
    fp=task1_cm['fp'] if task1_cm else None,
    tn=task1_cm['tn'] if task1_cm else None,
    fn=task1_cm['fn'] if task1_cm else None,
    predict_modeling_dataset=fincausal_task1_predict_modeling_dataset,
    train_file_path=Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/train.csv'),
    val_file_path=Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/dev.csv'),
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/145_1/checkpoint-1116-epoch-1/inference_test')
)
