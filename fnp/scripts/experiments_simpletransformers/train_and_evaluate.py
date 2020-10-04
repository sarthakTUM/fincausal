from pathlib import Path

from fnp.fincausal import utils
from fnp.fincausal.evaluation.inference import FinCausalTask1Inference
from fnp.fincausal.evaluation.metrics import MetricsWrapper
from fnp.fincausal.model.simpletransformers_model import SimpleTransformersModel
from fnp.fincausal.preprocessing.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter

############################################################################################

# 1. specify paths

"""
Specify the paths required for running this script:
1. train_file_path: Path to CSV for FinCausal training data
2. val_file_path: Path to CSV for FinCausal validation data
3. test_file_path: Path to CSV for FinCausal test data
4. pretrained_model_path: Path to the pre-trained transformers model
5. output_model_path: Root path to output all the training artifacts like trained models, predictions, metrics, etc.
"""

fincausal_task1_train_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/train.csv')
fincausal_task1_val_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/dev.csv')
fincausal_task1_test_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/test.csv')
pretrained_model_path = Path('/media/sarthak/HDD/data_science/fnp_resources/pretrained_models/finbert_latest')
output_model_path = Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/145_1')

############################################################################################

# 2. load the file using FileReader

"""
This step reads the raw dataset from the CSV files and converts to a standardized data type for further processing
"""

fincausal_task1_train_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_train_file_path)
fincausal_task1_val_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_val_file_path)
fincausal_task1_test_dataset_csv_reader = FinCausalTask1DatasetCSVAdapter(fincausal_task1_test_file_path)

fincausal_task1_train_dataset = fincausal_task1_train_dataset_csv_reader.read()
fincausal_task1_val_dataset = fincausal_task1_val_dataset_csv_reader.read()
fincausal_task1_test_dataset = fincausal_task1_test_dataset_csv_reader.read()
print('train: {}, val: {}, test: {}'.format(len(fincausal_task1_train_dataset),
                                            len(fincausal_task1_val_dataset),
                                            len(fincausal_task1_test_dataset)))

############################################################################################

# 3. create modeling instances from the dataset_instances and the feature_extractors

"""
This step takes the raw dataset from previous step and extract rules from them. Since we do not require any
rules for the BERT based classifiers, the list is empty. For example on how to use these, check 
`experiments_rules/train_and_evaluate.py` file. 
"""

feature_extractors = []  # For the possible feature extractors, check: `data_types/feature_extractors.py`

fincausal_task1_train_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_train_dataset,
                                                                                     feature_extractors=feature_extractors)
fincausal_task1_val_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_val_dataset,
                                                                                   feature_extractors=feature_extractors)
fincausal_task1_test_modeling_dataset_reader = FinCausalTask1ModelingDatasetAdapter(fincausal_task1_dataset=fincausal_task1_test_dataset,
                                                                                    feature_extractors=feature_extractors)

fincausal_task1_train_modeling_dataset = fincausal_task1_train_modeling_dataset_reader.read()
fincausal_task1_val_modeling_dataset = fincausal_task1_val_modeling_dataset_reader.read()
fincausal_task1_test_modeling_dataset = fincausal_task1_test_modeling_dataset_reader.read()

#############################################################################################

# 4 Create a Model (check names for HF models here: https://huggingface.co/transformers/pretrained_models.html

"""
This step loads the pre-trained transformers model from the local specified path (or directly specify the
path from the HuggingFace repository. The models are trained using the SimpleTransformers library which
is a wrapper over the Transformers library.
"""

imbalanced_learning = True  # for imbalanced classification settings, the losses are assigned the weights

simpletransformers_model = SimpleTransformersModel(
    model_type='bert',  # possible to replace this with another architecture like RoBERTa, etc.
    model_name_or_path=pretrained_model_path,  # can be local file path or huggingface repository path
    output_dir=output_model_path,  # path for storing the training related artifacts
    class_weights=None if not imbalanced_learning else utils.calculate_class_weights_task1(fincausal_task1_train_modeling_dataset))


##############################################################################################

# 5. fit the model to the data

"""
This section fits the model on the training data and validates on validation data. Specify the 
parameters specific to training, such as, epoch numbers, etc.
"""

simpletransformers_model.fit_and_evaluate(train_dataset=fincausal_task1_train_modeling_dataset,
                                          val_dataset=fincausal_task1_val_modeling_dataset,
                                          num_train_epochs=3,
                                          max_seq_length=200,
                                          train_batch_size=8,
                                          gradient_accumulation_steps=2,
                                          random_oversample=False)

###############################################################################################

# 6. Predict on the test dataset

"""
This section loads the best model from the epochs above and makes the predictions on the test dataset
"""

# 6.1 Initialize the model
simpletransformers_model = SimpleTransformersModel(
    model_type='bert',
    model_name_or_path=Path(output_model_path / 'output/best_model'),
    output_dir=Path(output_model_path / 'output/best_model'))

# 6.2 Make the prediction
simpletransformers_model.predict_on_dataset(dataset=fincausal_task1_test_modeling_dataset)

################################################################################################

# 7. evaluate the predictions

"""
This section takes the predictions generated and runs the evaluation to calaulate the metrics
"""

# 7.1 Initialize the metric calculators
task1_metrics = MetricsWrapper.calculate_metrics_task1(dataset=fincausal_task1_test_modeling_dataset)
task1_cm = MetricsWrapper.calculate_confusionmatrix_task1(dataset=fincausal_task1_test_modeling_dataset)
print(task1_metrics)
print(task1_cm)

# 7.2 Save the metrics and final predictions to files
fincausal_task1_inference = FinCausalTask1Inference(
    model_path=Path(output_model_path / 'output/best_model'),
    predict_file_path=fincausal_task1_test_file_path,
    f1=task1_metrics['f1'],
    recall=task1_metrics['recall'],
    precision=task1_metrics['precision'],
    tp=task1_cm['tp'],
    fp=task1_cm['fp'],
    tn=task1_cm['tn'],
    fn=task1_cm['fn'],
    predict_modeling_dataset=fincausal_task1_test_modeling_dataset,
    train_file_path=fincausal_task1_train_file_path,
    val_file_path=fincausal_task1_val_file_path,
    output_dir=Path(output_model_path / 'output/best_model/inference')
)

#################################################################################################

