from pathlib import Path
from fnp.fincausal.evaluation.inference import FinCausalTask1Inference
from fnp.fincausal.evaluation.metrics import MetricsWrapper
from fnp.fincausal.model.simpletransformers_model import SimpleTransformersModel
from fnp.fincausal.preprocessing.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter

############################################################################################

# 1. specify paths

"""
Specify the paths required for running this script:
1. fincausal_task1_predict_file_path: Path to CSV for FinCausal prediction data
2. model_path: Path to the pre-trained transformers model
3. inference_artifacts_file_path: Root path to output all the training artifacts like trained models, predictions, metrics, etc.
"""

fincausal_task1_predict_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/test.csv')
model_path = Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/145_1')
inference_artifacts_file_path = model_path / 'inference_test'

############################################################################################

# 2. load the file using FileReader

"""
This step reads the raw dataset from the CSV files and converts to a standardized data type for further processing
"""

fincausal_task1_predict_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_predict_file_path)
fincausal_task1_predict_dataset = fincausal_task1_predict_dataset_csv_reader.read()

print('predict: {}'.format(len(fincausal_task1_predict_dataset)))

############################################################################################

# 3. create modeling instances from the dataset_instances and the feature_extractors

"""
This step takes the raw dataset from previous step and extract rules from them. Since we do not require any
rules for the BERT based classifiers, the list is empty. For example on how to use these, check 
`experiments_rules/train_and_evaluate.py` file. 
"""

feature_extractors = []  # For the possible feature extractors, check: `data_types/feature_extractors.py`

fincausal_task1_predict_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_predict_dataset,
                                                                                       feature_extractors=feature_extractors)

fincausal_task1_predict_modeling_dataset = fincausal_task1_predict_modeling_dataset_reader.read()

##############################################################################################

# 4 Predict on the test dataset

"""
This section loads the best model from the epochs above and makes the predictions on the prediction dataset
"""

# 4.1 Initialize the model
simpletransformers_model = SimpleTransformersModel(
    model_type='bert',
    model_name_or_path=model_path,
    output_dir=model_path)

# 4.2 Make the prediction
simpletransformers_model.predict_on_dataset(dataset=fincausal_task1_predict_modeling_dataset)

################################################################################################

# 5. evaluate the predictions

"""
This section takes the predictions generated and runs the evaluation to calculate the metrics
"""

# 5.1 Initialize the metric calculators
if fincausal_task1_predict_modeling_dataset.is_labeled:
    task1_metrics = MetricsWrapper.calculate_metrics_task1(dataset=fincausal_task1_predict_modeling_dataset)
    task1_cm = MetricsWrapper.calculate_confusionmatrix_task1(dataset=fincausal_task1_predict_modeling_dataset)
    print(task1_metrics)
    print(task1_cm)
else:
    task1_metrics = None
    task1_cm = None

# 5.2 Save the metrics and final predictions to files
fincausal_task1_inference = FinCausalTask1Inference(
    model_path=model_path,
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
    output_dir=inference_artifacts_file_path)

#################################################################################################

