from pathlib import Path

from fnp.fincausal import utils
from fnp.fincausal.data_types.dataset_reader import FinCausalTask1DatasetCSVAdapter, \
    FinCausalTask1ModelingDatasetAdapter
from fnp.fincausal.data_types.feature_extractors import ContainsCausalConnectiveFeatureExtractor, \
    ContainsNumericFeatureExtractor, ContainsPercentFeatureExtractor, ContainsCurrencyFeatureExtractor, \
    ContainsTextualNumericFeatureExtractor, ContainsVerbAfterCommaFeatureExtractor, \
    ContainsSpecificVerbAfterCommaFeatureExtractor, POSofRootFeatureExtractor

# 1. specify a file to read
from fnp.fincausal.data_types.model import SklearnRandomForest, XGBoostClassifier
from fnp.fincausal.evaluation.inference import FinCausalTask1Inference
from fnp.fincausal.evaluation.metrics import MetricsWrapper

fincausal_task1_train_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/train.csv')
fincausal_task1_val_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/dev.csv')
fincausal_task1_test_file_path = Path('/media/sarthak/HDD/data_science/fnp_resources/data/task1/all_combined/test.csv')
fincausal_task1_model_path = Path('/media/sarthak/HDD/data_science/fnp_resources/fincausal_t1_models/113_1/output/best_model')

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
feature_extractors = [ContainsCausalConnectiveFeatureExtractor(causal_connectives=['as',
                                                                                   'since',
                                                                                   'because',
                                                                                   'cause',
                                                                                   'after']),
                      ContainsNumericFeatureExtractor(),
                      ContainsPercentFeatureExtractor(),
                      ContainsCurrencyFeatureExtractor(currencies=['$', '€', '£', 'yuan', 'Yuan', 'INR', 'inr']),
                      ContainsSpecificVerbAfterCommaFeatureExtractor(verbs=[', reaching',
                                                                            ', prompting',
                                                                            ', aiming',
                                                                            ', equating',
                                                                            ', hitting',
                                                                            ', lowering',
                                                                            ', topping',
                                                                            ', raising',
                                                                            ', converting',
                                                                            ', becoming',
                                                                            ', meeting',
                                                                            ', valuing',
                                                                            ', edging',
                                                                            ', boosting',
                                                                            ', completing',
                                                                            ', slowing',
                                                                            ', lasting',
                                                                            ', clothing',
                                                                            ', totaling',
                                                                            ', rising']),
                      ContainsTextualNumericFeatureExtractor(textual_numerics=['one', 'two', 'three', 'four', 'five',
                                                                               'six', 'seven', 'eight', 'nine', 'ten']),
                      ContainsVerbAfterCommaFeatureExtractor(regex=""",\s([a-z]*?ing)"""),
                      POSofRootFeatureExtractor()]

# 4. create modeling instances from the dataset_instances and the feature_extractors
print('extracting features')
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
classifier = XGBoostClassifier(output_dir=fincausal_task1_model_path,
                               reload=False,
                               class_weights=None)

# 6. Fit the model to the data
print('training model')
classifier.fit_and_evaluate(train_dataset=fincausal_task1_train_modeling_dataset,
                            val_dataset=fincausal_task1_val_modeling_dataset)


# 7. Predict on the test dataset
print('predicting..')
classifier = XGBoostClassifier(output_dir=fincausal_task1_model_path,
                               reload=True)
classifier.predict_on_dataset(dataset=fincausal_task1_test_modeling_dataset)

# 8. evaluate the predictions
task1_metrics = MetricsWrapper.calculate_metrics_task1(dataset=fincausal_task1_test_modeling_dataset)
task1_cm = MetricsWrapper.calculate_confusionmatrix_task1(dataset=fincausal_task1_test_modeling_dataset)
print(task1_metrics)
print(task1_cm)

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
