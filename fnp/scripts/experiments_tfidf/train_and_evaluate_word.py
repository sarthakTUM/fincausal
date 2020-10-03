from pathlib import Path

# 1. specify a file to read
from fnp.fincausal.evaluation.inference import FinCausalTask1Inference
from fnp.fincausal.evaluation.metrics import MetricsWrapper
from fnp.fincausal.model.sklearn_model import TfIdfVectorizerXgbClassifier
from fnp.fincausal.preprocessing.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter

fincausal_task1_train_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/train_on_trial_test_on_practice_v2/train.csv')
fincausal_task1_val_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/train_on_trial_test_on_practice_v2/dev.csv')
fincausal_task1_test_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/train_on_trial_test_on_practice_v2/test.csv')
fincausal_task1_model_path = Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/91_1/output/best_model')

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


# 5. Create a Model
classifier = TfIdfVectorizerXgbClassifier(output_dir=fincausal_task1_model_path,
                                          reload=False)

# 6. Fit the model to the data
classifier.fit_and_evaluate(train_dataset=fincausal_task1_train_modeling_dataset,
                            val_dataset=fincausal_task1_val_modeling_dataset)


# 7. Predict on the test dataset
classifier = TfIdfVectorizerXgbClassifier(output_dir=fincausal_task1_model_path,
                                          reload=True)
classifier.predict_on_dataset(dataset=fincausal_task1_test_modeling_dataset)

# 8. evaluate the predictions
task1_metrics = MetricsWrapper.calculate_metrics_task1(dataset=fincausal_task1_test_modeling_dataset)
task1_cm = MetricsWrapper.calculate_confusionmatrix_task1(dataset=fincausal_task1_test_modeling_dataset)
print('test_precision: {} test_recall: {} test_f1: {}'.format(task1_metrics['precision'],
                                                              task1_metrics['recall'],
                                                              task1_metrics['f1']))
print('test confusion matrix: {}'.format(task1_cm))

fincausal_task1_inference = FinCausalTask1Inference(
    model_path=Path(fincausal_task1_model_path),
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
    output_dir=Path(fincausal_task1_model_path / 'inference')
)
