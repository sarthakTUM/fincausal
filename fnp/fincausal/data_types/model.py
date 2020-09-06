from typing import List, Tuple, Dict, Optional, Union

from pathlib import Path

from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from hpsklearn import HyperoptEstimator, xgboost_classification
from sklearn import metrics
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from fnp.fincausal.data_types.core import Model, FinCausalDataset
from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset
from fnp.fincausal.data_types.features import BooleanFeature, OneHotFeature
from fnp.fincausal.data_types.label import FinCausalTask1Pred
from fnp.fincausal.data_types.modeling_instance import FinCausalTask1ModelingInstance
from simpletransformers.classification import ClassificationModel
from fnp.fincausal import utils
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

import numpy as np
import pandas as pd
from scipy.sparse import hstack

from xgboost import XGBClassifier

from fnp.fincausal.evaluation.metrics import Metrics


class SimpleTransformersModel(Model):
    def __init__(self,
                 model_type: str,
                 model_name_or_path: Union[str, Path],
                 output_dir: Path,
                 class_weights: Optional[List[float]] = None
                 ):
        print('class weights: {}'.format(class_weights))
        self.output_dir = output_dir
        self.cache_dir = output_dir / 'cache/'
        self.tensorboard_dir = output_dir / 'runs/'
        self.best_model_dir = output_dir / 'output/best_model/'

        self.model_type = model_type
        self.model_name_or_path = model_name_or_path

        self.model = ClassificationModel(self.model_type,
                                         str(self.model_name_or_path),
                                         cache_dir='/media/sarthak/HDD/data_science/fnp_resources/pretrained_models/',
                                         args={'fp_16': True,
                                               'output_dir': str(self.output_dir),
                                               'cache_dir': str(self.cache_dir),
                                               'tensorboard_dir': str(self.tensorboard_dir),
                                               'best_model_dir': str(self.best_model_dir)},
                                         weight=class_weights
                                         )

        self.class_weights = class_weights

    def fit(self,
            train_dataset: FinCausalTask1ModelingDataset,
            num_train_epochs: Optional[int] = 3):
        train_df = SimpleTransformersModel._convert_modeling_dataset_to_training_dataset(train_dataset)
        self.model.train_model(train_df,
                               show_running_loss=False,
                               args={
                                   'num_train_epochs': num_train_epochs
                               })

    def fit_and_evaluate(self,
                         train_dataset: FinCausalTask1ModelingDataset,
                         val_dataset: FinCausalTask1ModelingDataset,
                         num_train_epochs: Optional[int] = 3,
                         train_batch_size: Optional[int] = 8,
                         max_seq_length: Optional[int] = 128,
                         gradient_accumulation_steps: Optional[int] = 1,
                         random_oversample: bool=False):
        train_df = SimpleTransformersModel._convert_modeling_dataset_to_training_dataset(train_dataset)
        eval_df = SimpleTransformersModel._convert_modeling_dataset_to_training_dataset(val_dataset)
        if random_oversample:
            ros = RandomOverSampler(random_state=42)
            train_df_x, train_df_y = ros.fit_resample(train_df['text'].values[..., np.newaxis], train_df['labels'])
            train_df = pd.DataFrame({
                'text': train_df_x.flatten(),
                'labels': train_df_y
            })
            train_df = train_df.sample(n=len(train_df))
        self.model.train_model(train_df=train_df,
                               eval_df=eval_df,
                               show_running_loss=False,
                               args={
                                   'evaluate_during_training': False,
                                   'evaluate_after_every_epoch': True,
                                   'evaluate_after_every_epoch_verbose': True,
                                   'use_cached_eval_features': True,
                                   'num_train_epochs': num_train_epochs,
                                   'train_batch_size': train_batch_size,
                                   'early_stopping_metric': 'f1',
                                   'early_stopping_metric_minimize': False,
                                   'save_during_training': False,
                                   'use_early_stopping': True,
                                   'early_stopping_patience': 2,
                                   'max_seq_length': max_seq_length,
                                   'gradient_accumulation_steps': gradient_accumulation_steps
                               },
                               f1=Metrics.weighted_f1_score)

    @staticmethod
    def _convert_modeling_dataset_to_training_dataset(dataset: FinCausalTask1ModelingDataset) -> pd.DataFrame:
        texts = []
        labels = []
        for modeling_instance in dataset:
            text = modeling_instance.dataset_instance.text
            label = modeling_instance.label.causal
            texts.append(text)
            labels.append(label)
        assert len(texts) == len(labels)
        return pd.DataFrame({
            'text': texts,
            'labels': labels
        })

    def predict_on_dataset(self, dataset: FinCausalTask1ModelingDataset):
        to_predict_texts = []
        for modeling_instance in dataset:
            to_predict_texts.append(modeling_instance.dataset_instance.text)
        preds, logits = self.model.predict(to_predict=to_predict_texts)
        for modeling_instance, pred, logit in zip(dataset, preds, logits):
            modeling_instance.pred = FinCausalTask1Pred(causal=pred, confidence=max(utils.softmax(logit)))


class SklearnRandomForest(Model):

    def fit(self, dataset: FinCausalDataset, num_train_epochs: Optional[int] = None):
        pass

    def __init__(self,
                 output_dir: Path,
                 reload: bool,
                 class_weights: Optional[str] = None
                 ):
        self.reload = reload
        self.output_dir = output_dir
        if self.reload:
            reload_path = self.output_dir / 'model.joblib'
            self.model = joblib.load(reload_path)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self.model = RandomForestClassifier(max_depth=100,
                                                n_estimators=1000,
                                                random_state=0,
                                                verbose=1,
                                                class_weight=class_weights)

    def fit_and_evaluate(self,
                         train_dataset: FinCausalTask1ModelingDataset,
                         val_dataset: FinCausalTask1ModelingDataset):
        """
        This function should do everything to train the model. In case of SkLearn models, it is a simple fit.
        In case of a deep learning model, this should run a batcher and run epochs. Ideally, the modeling_instances
        should be converted to the training_instances before passing to the model. Since every model has a different
        input requirement, it makes sense to let this function convert the modeling_instance to the training_instance
        iself.
        :param val_dataset:
        :param train_dataset:
        :return:
        """

        # convert the modeling_instances into input suitable for LogReg model
        train_X, train_y = SklearnRandomForest._convert_modeling_dataset_to_training_dataset(train_dataset)
        val_X, val_y = SklearnRandomForest._convert_modeling_dataset_to_training_dataset(val_dataset)
        print('fitting')
        self.model.fit(train_X, train_y)
        val_y_predict = self.model.predict(val_X)
        train_y_predict = self.model.predict(train_X)
        metrics_val = metrics.precision_recall_fscore_support(val_y, val_y_predict, average='weighted')
        metrics_train = metrics.precision_recall_fscore_support(train_y, train_y_predict, average='weighted')
        joblib.dump(self.model, self.output_dir / 'model.joblib')
        print("""val_precision: {} val_recall: {} val_f1: {}\n
                 train_precision: {} train_recall: {} train_f1: {}""".format(metrics_val[0],
                                                                             metrics_val[1],
                                                                             metrics_val[2],
                                                                             metrics_train[0],
                                                                             metrics_train[1],
                                                                             metrics_train[2]))

    def predict_on_dataset(self, dataset: FinCausalTask1ModelingDataset):
        X, y = SklearnRandomForest._convert_modeling_dataset_to_training_dataset(dataset)
        y_pred = self.model.predict_proba(X)
        for modeling_instance, pred in zip(dataset, y_pred):
            modeling_instance.pred = FinCausalTask1Pred(causal=np.argmax(pred), confidence=pred[np.argmax(pred)])  # ToDo change the conf. score

    @staticmethod
    def _convert_modeling_dataset_to_training_dataset(modeling_dataset: FinCausalTask1ModelingDataset) -> Tuple[
        np.array, np.array]:
        features_for_each_instance = []
        label_for_each_instance = []
        for modeling_instance in modeling_dataset.instances:
            features_for_each_instance.append(
                SklearnRandomForest._convert_modeling_instance_to_training_instance(modeling_instance))
            label_for_each_instance.append(
                SklearnRandomForest._extract_label_for_modeling_instance_(modeling_instance))
        return np.array(features_for_each_instance), np.array(label_for_each_instance).ravel()

    @staticmethod
    def _convert_modeling_instance_to_training_instance(modeling_instance: FinCausalTask1ModelingInstance) -> List[bool]:

        features = []
        if modeling_instance.features:
            for feature in modeling_instance.features:
                if isinstance(feature, BooleanFeature):
                    features.append(feature.boolean_value)
                elif isinstance(feature, OneHotFeature):
                    features.extend(feature.one_hot)

        return features

    @staticmethod
    def _extract_label_for_modeling_instance_(modeling_instance: FinCausalTask1ModelingInstance) -> int:
        return modeling_instance.label.causal


class XGBoostClassifier(Model):

    def fit(self, dataset: FinCausalDataset, num_train_epochs: Optional[int] = None):
        pass

    def __init__(self,
                 output_dir: Path,
                 reload: bool,
                 class_weights: Optional[List[float]] = None
                 ):
        self.reload = reload
        self.output_dir = output_dir
        if self.reload:
            reload_path = self.output_dir / 'model.joblib'
            self.model = joblib.load(reload_path)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self.model = XGBClassifier(n_estimators=150,
                                       max_depth=5,
                                       min_child_weight=1,
                                       learning_rate=0.1,
                                       verbosity=1)

    def fit_and_evaluate(self,
                         train_dataset: FinCausalTask1ModelingDataset,
                         val_dataset: FinCausalTask1ModelingDataset):
        """
        This function should do everything to train the model. In case of SkLearn models, it is a simple fit.
        In case of a deep learning model, this should run a batcher and run epochs. Ideally, the modeling_instances
        should be converted to the training_instances before passing to the model. Since every model has a different
        input requirement, it makes sense to let this function convert the modeling_instance to the training_instance
        iself.
        :param val_dataset:
        :param train_dataset:
        :return:
        """

        # convert the modeling_instances into input suitable for LogReg model
        train_X, train_y = XGBoostClassifier._convert_modeling_dataset_to_training_dataset(train_dataset)
        val_X, val_y = XGBoostClassifier._convert_modeling_dataset_to_training_dataset(val_dataset)
        print('fitting')
        self.model.fit(train_X, train_y, early_stopping_rounds=10, eval_set=[(val_X, val_y)], eval_metric="error", verbose=True)
        val_y_predict = self.model.predict(val_X)
        train_y_predict = self.model.predict(train_X)
        metrics_val = metrics.precision_recall_fscore_support(val_y, val_y_predict, average='weighted')
        metrics_train = metrics.precision_recall_fscore_support(train_y, train_y_predict, average='weighted')
        joblib.dump(self.model, self.output_dir / 'model.joblib')
        print("""val_precision: {} val_recall: {} val_f1: {}\n
                 train_precision: {} train_recall: {} train_f1: {}""".format(metrics_val[0],
                                                                             metrics_val[1],
                                                                             metrics_val[2],
                                                                             metrics_train[0],
                                                                             metrics_train[1],
                                                                             metrics_train[2]))

    def predict_on_dataset(self, dataset: FinCausalTask1ModelingDataset):
        X, y = XGBoostClassifier._convert_modeling_dataset_to_training_dataset(dataset)
        y_pred = self.model.predict_proba(X)
        for modeling_instance, pred in zip(dataset, y_pred):
            modeling_instance.pred = FinCausalTask1Pred(causal=np.argmax(pred), confidence=pred[np.argmax(pred)])  # ToDo change the conf. score

    @staticmethod
    def _convert_modeling_dataset_to_training_dataset(modeling_dataset: FinCausalTask1ModelingDataset) -> Tuple[
        np.array, np.array]:
        features_for_each_instance = []
        label_for_each_instance = []
        for modeling_instance in modeling_dataset.instances:
            features_for_each_instance.append(
                XGBoostClassifier._convert_modeling_instance_to_training_instance(modeling_instance))
            label_for_each_instance.append(
                XGBoostClassifier._extract_label_for_modeling_instance_(modeling_instance))
        return np.array(features_for_each_instance), np.array(label_for_each_instance).ravel()

    @staticmethod
    def _convert_modeling_instance_to_training_instance(modeling_instance: FinCausalTask1ModelingInstance) -> List[bool]:

        features = []
        if modeling_instance.features:
            for feature in modeling_instance.features:
                if isinstance(feature, BooleanFeature):
                    features.append(feature.boolean_value)
                elif isinstance(feature, OneHotFeature):
                    features.extend(feature.one_hot)

        return features

    @staticmethod
    def _extract_label_for_modeling_instance_(modeling_instance: FinCausalTask1ModelingInstance) -> int:
        return modeling_instance.label.causal


class XGBoostClassifierHP(Model):

    def fit(self, dataset: FinCausalDataset, num_train_epochs: Optional[int] = None):
        pass

    def __init__(self,
                 output_dir: Path,
                 reload: bool,
                 class_weights: Optional[List[float]] = None
                 ):
        self.reload = reload
        self.output_dir = output_dir
        self.space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                 'gamma': hp.uniform('gamma', 1, 9),
                 'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                 'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                 'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                 'n_estimators': 180,
                 'seed': 0
                 }
        if self.reload:
            reload_path = self.output_dir / 'model.joblib'
            self.model = joblib.load(reload_path)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)

    def fit_and_evaluate(self,
                         train_dataset: FinCausalTask1ModelingDataset,
                         val_dataset: FinCausalTask1ModelingDataset):
        """
        This function should do everything to train the model. In case of SkLearn models, it is a simple fit.
        In case of a deep learning model, this should run a batcher and run epochs. Ideally, the modeling_instances
        should be converted to the training_instances before passing to the model. Since every model has a different
        input requirement, it makes sense to let this function convert the modeling_instance to the training_instance
        iself.
        :param val_dataset:
        :param train_dataset:
        :return:
        """

        # convert the modeling_instances into input suitable for LogReg model
        self.train_X, self.train_y = XGBoostClassifier._convert_modeling_dataset_to_training_dataset(train_dataset)
        self.val_X, self.val_y = XGBoostClassifier._convert_modeling_dataset_to_training_dataset(val_dataset)

        trials = Trials()

        best_hyperparams = fmin(fn=self._optimize,
                                space=self.space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)

    def _optimize(self, params):
        model = XGBClassifier(n_estimators=params['n_estimators'],
                                   max_depth=int(params['max_depth']),
                                   gamma=params['gamma'],
                                   reg_alpha=int(params['reg_alpha']),
                                   min_child_weight=int(params['min_child_weight']),
                                   colsample_bytree=int(params['colsample_bytree']))
        evaluation = [(self.train_X, self.train_y), (self.val_X, self.val_y)]
        model.fit(self.train_X,
                       self.train_y,
                       eval_set=evaluation,
                       eval_metric="auc",
                       early_stopping_rounds=10,
                       verbose=False)
        val_y_predict = model.predict(self.val_X)
        train_y_predict = model.predict(self.train_X)
        metrics_val = metrics.precision_recall_fscore_support(self.val_y, val_y_predict, average='weighted')
        metrics_train = metrics.precision_recall_fscore_support(self.train_y, train_y_predict, average='weighted')
        print("""val_precision: {} val_recall: {} val_f1: {}\n
                 train_precision: {} train_recall: {} train_f1: {},
                 'status': {}""".format(metrics_val[0],
                                        metrics_val[1],
                                        metrics_val[2],
                                        metrics_train[0],
                                        metrics_train[1],
                                        metrics_train[2],
                                        STATUS_OK))

    def predict_on_dataset(self, dataset: FinCausalTask1ModelingDataset):
        X, y = XGBoostClassifier._convert_modeling_dataset_to_training_dataset(dataset)
        y_pred = self.model.predict(X)
        for modeling_instance, pred in zip(dataset, y_pred):
            modeling_instance.pred = FinCausalTask1Pred(causal=pred)  # ToDo change the conf. score

    @staticmethod
    def _convert_modeling_dataset_to_training_dataset(modeling_dataset: FinCausalTask1ModelingDataset) -> Tuple[
        np.array, np.array]:
        features_for_each_instance = []
        label_for_each_instance = []
        for modeling_instance in modeling_dataset.instances:
            features_for_each_instance.append(
                XGBoostClassifier._convert_modeling_instance_to_training_instance(modeling_instance))
            label_for_each_instance.append(
                XGBoostClassifier._extract_label_for_modeling_instance_(modeling_instance))
        return np.array(features_for_each_instance), np.array(label_for_each_instance).ravel()

    @staticmethod
    def _convert_modeling_instance_to_training_instance(modeling_instance: FinCausalTask1ModelingInstance) -> List[bool]:

        features = []
        if modeling_instance.features:
            for feature in modeling_instance.features:
                if isinstance(feature, BooleanFeature):
                    features.append(feature.boolean_value)

        return features

    @staticmethod
    def _extract_label_for_modeling_instance_(modeling_instance: FinCausalTask1ModelingInstance) -> int:
        return modeling_instance.label.causal


class TfIdfVectorizerXgbClassifier(Model):

    def fit(self, dataset: FinCausalDataset):
        pass

    def __init__(self,
                 output_dir: Path,
                 reload: bool,
                 class_weights: Optional[List[float]] = None
                 ):
        self.vectorizer = TfidfVectorizer(
                                            sublinear_tf=True,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            ngram_range=(1, 1),
                                            norm='l2',
                                            min_df=0,
                                            smooth_idf=False,
                                            max_features=15000
                                         )
        self.reload = reload
        self.output_dir = output_dir
        if self.reload:
            model_reload_path = self.output_dir / 'model.joblib'
            vectorizer_reload_path = self.output_dir / 'tf_idf_vectorizer_fitted.joblib'
            self.model = joblib.load(model_reload_path)
            self.vectorizer = joblib.load(vectorizer_reload_path)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self.model = XGBClassifier(n_estimators=200,
                                       max_depth=2000,
                                       learning_rate=0.3,
                                       verbosity=1,
                                       scale_pos_weight=9)

    def fit_and_evaluate(self,
                         train_dataset: FinCausalTask1ModelingDataset,
                         val_dataset: FinCausalTask1ModelingDataset):
        tf_idf_vectorizer_fitted = self.vectorizer.fit([modeling_instance.dataset_instance.text
                                                        for modeling_instance in train_dataset])
        train_X, train_y = TfIdfVectorizerXgbClassifier._convert_modeling_dataset_to_training_dataset(train_dataset,
                                                                                                      tf_idf_vectorizer_fitted)
        val_X, val_y = TfIdfVectorizerXgbClassifier._convert_modeling_dataset_to_training_dataset(val_dataset,
                                                                                                  tf_idf_vectorizer_fitted)
        print('fitting')
        self.model.fit(train_X, train_y)
        val_y_predict = self.model.predict(val_X)
        train_y_predict = self.model.predict(train_X)
        metrics_val = metrics.precision_recall_fscore_support(val_y, val_y_predict, average='weighted')
        metrics_train = metrics.precision_recall_fscore_support(train_y, train_y_predict, average='weighted')
        joblib.dump(self.model, self.output_dir / 'model.joblib')
        joblib.dump(tf_idf_vectorizer_fitted, self.output_dir / 'tf_idf_vectorizer_fitted.joblib')
        print("""val_precision: {} val_recall: {} val_f1: {} \ntrain_precision: {} train_recall: {} train_f1: {}""".format(metrics_val[0],
                                                                                     metrics_val[1],
                                                                                     metrics_val[2],
                                                                                     metrics_train[0],
                                                                                     metrics_train[1],
                                                                                     metrics_train[2]))

    def predict_on_dataset(self, dataset: FinCausalTask1ModelingDataset):
        X, y = TfIdfVectorizerXgbClassifier._convert_modeling_dataset_to_training_dataset(dataset, self.vectorizer)
        y_pred = self.model.predict(X)
        for modeling_instance, pred in zip(dataset, y_pred):
            modeling_instance.pred = FinCausalTask1Pred(causal=pred)  # ToDo change the conf. score

    @staticmethod
    def _convert_modeling_dataset_to_training_dataset(modeling_dataset: FinCausalTask1ModelingDataset,
                                                      tf_idf_vectorizer_fitted):

        features_for_each_instance = tf_idf_vectorizer_fitted.transform(
            [modeling_instance.dataset_instance.text for modeling_instance in modeling_dataset])

        label_for_each_instance = [modeling_instance.label.causal for modeling_instance in modeling_dataset]

        return features_for_each_instance, np.array(label_for_each_instance).ravel()


class TfIdfWordCharVectorizerXgbClassifier(Model):

    def fit(self, dataset: FinCausalDataset):
        pass

    def __init__(self,
                 output_dir: Path,
                 reload: bool,
                 class_weights: Optional[List[float]] = None
                 ):
        self.word_vectorizer = TfidfVectorizer(
                                            sublinear_tf=True,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            ngram_range=(1, 1),
                                            norm='l2',
                                            min_df=0,
                                            smooth_idf=False,
                                            max_features=15000
                                         )
        self.char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            ngram_range=(1, 1),
            norm='l2',
            min_df=0,
            smooth_idf=False,
            max_features=50000
        )
        self.reload = reload
        self.output_dir = output_dir
        if self.reload:
            model_reload_path = self.output_dir / 'model.joblib'
            word_vectorizer_reload_path = self.output_dir / 'tf_idf_word_vectorizer_fitted.joblib'
            char_vectorizer_reload_path = self.output_dir / 'tf_idf_char_vectorizer_fitted.joblib'
            self.model = joblib.load(model_reload_path)
            self.word_vectorizer = joblib.load(word_vectorizer_reload_path)
            self.char_vectorizer = joblib.load(char_vectorizer_reload_path)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self.model = XGBClassifier(n_estimators=100,
                                       max_depth=10,
                                       learning_rate=0.3,
                                       verbosity=1)

    def fit_and_evaluate(self,
                         train_dataset: FinCausalTask1ModelingDataset,
                         val_dataset: FinCausalTask1ModelingDataset):
        tf_idf_word_vectorizer_fitted = self.word_vectorizer.fit([modeling_instance.dataset_instance.text
                                                        for modeling_instance in train_dataset])
        tf_idf_char_vectorizer_fitted = self.char_vectorizer.fit([modeling_instance.dataset_instance.text
                                                                  for modeling_instance in train_dataset])
        train_X, train_y = TfIdfWordCharVectorizerXgbClassifier._convert_modeling_dataset_to_training_dataset(train_dataset,
                                                                                                              tf_idf_word_vectorizer_fitted,
                                                                                                              tf_idf_char_vectorizer_fitted)
        val_X, val_y = TfIdfWordCharVectorizerXgbClassifier._convert_modeling_dataset_to_training_dataset(val_dataset,
                                                                                                          tf_idf_word_vectorizer_fitted,
                                                                                                          tf_idf_char_vectorizer_fitted)
        print('fitting')
        self.model.fit(train_X, train_y)
        val_y_predict = self.model.predict(val_X)
        train_y_predict = self.model.predict(train_X)
        metrics_val = metrics.precision_recall_fscore_support(val_y, val_y_predict, average='weighted')
        metrics_train = metrics.precision_recall_fscore_support(train_y, train_y_predict, average='weighted')
        joblib.dump(self.model, self.output_dir / 'model.joblib')
        joblib.dump(tf_idf_word_vectorizer_fitted, self.output_dir / 'tf_idf_word_vectorizer_fitted.joblib')
        joblib.dump(tf_idf_char_vectorizer_fitted, self.output_dir / 'tf_idf_char_vectorizer_fitted.joblib')
        print("""val_precision: {} val_recall: {} val_f1: {} \ntrain_precision: {} train_recall: {} train_f1: {}""".format(metrics_val[0],
                                                                                     metrics_val[1],
                                                                                     metrics_val[2],
                                                                                     metrics_train[0],
                                                                                     metrics_train[1],
                                                                                     metrics_train[2]))

    def predict_on_dataset(self, dataset: FinCausalTask1ModelingDataset):
        X, y = TfIdfWordCharVectorizerXgbClassifier._convert_modeling_dataset_to_training_dataset(dataset,
                                                                                                  self.word_vectorizer,
                                                                                                  self.char_vectorizer)
        y_pred = self.model.predict(X)
        for modeling_instance, pred in zip(dataset, y_pred):
            modeling_instance.pred = FinCausalTask1Pred(causal=pred)  # ToDo change the conf. score

    @staticmethod
    def _convert_modeling_dataset_to_training_dataset(modeling_dataset: FinCausalTask1ModelingDataset,
                                                      tf_idf_word_vectorizer_fitted,
                                                      tf_idf_char_vectorizer_fitted):

        word_features_for_each_instance = tf_idf_word_vectorizer_fitted.transform(
            [modeling_instance.dataset_instance.text for modeling_instance in modeling_dataset])
        char_features_for_each_instance = tf_idf_char_vectorizer_fitted.transform(
            [modeling_instance.dataset_instance.text for modeling_instance in modeling_dataset])

        features = hstack([word_features_for_each_instance, char_features_for_each_instance])

        label_for_each_instance = [modeling_instance.label.causal for modeling_instance in modeling_dataset]

        return features, np.array(label_for_each_instance).ravel()

