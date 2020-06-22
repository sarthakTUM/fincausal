from pathlib import Path

from fnp.fincausal import utils
from fnp.fincausal.data_types.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter
from fnp.fincausal.data_types.model import SimpleTransformersModel
from fnp.fincausal.evaluation.inference import FinCausalTask1Inference

from fnp.fincausal.evaluation.metrics import MetricsWrapper

# 1. specify a file to read
fincausal_task1_train_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/train_on_trial_test_on_practice_v2/train_dev.csv')
fincausal_task1_val_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/train_on_trial_test_on_practice_v2/test.csv')
fincausal_task1_test_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/train_on_trial_test_on_practice_v2/test.csv')

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
    model_type='bert',
    model_name_or_path='bert-base-uncased',
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/94_1'),
    class_weights=None)
    # class_weights=utils.calculate_class_weights_task1(finca+usal_task1_train_modeling_dataset))

# 6. fit the model to the data
simpletransformers_model.fit_and_evaluate(train_dataset=fincausal_task1_train_modeling_dataset,
                                          val_dataset=fincausal_task1_val_modeling_dataset,
                                          num_train_epochs=5,
                                          max_seq_length=200,
                                          train_batch_size=8,
                                          gradient_accumulation_steps=2)

# 7. Predict on the test dataset
simpletransformers_model = SimpleTransformersModel(
    model_type='bert',
    model_name_or_path=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/94_1/output/best_model'),
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/94_1/output/best_model'))

simpletransformers_model.predict_on_dataset(dataset=fincausal_task1_test_modeling_dataset)
# 8. evaluate the predictions
task1_metrics = MetricsWrapper.calculate_metrics_task1(dataset=fincausal_task1_test_modeling_dataset)
task1_cm = MetricsWrapper.calculate_confusionmatrix_task1(dataset=fincausal_task1_test_modeling_dataset)
print(task1_metrics)
print(task1_cm)

fincausal_task1_inference = FinCausalTask1Inference(
    model_path=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/94_1/output/best_model'),
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
    output_dir=Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/94_1/output/best_model/inference')
)

