from typing import List, Tuple, Dict, Optional, Union

from pathlib import Path
from sklearn.linear_model import LogisticRegression

from fnp.fincausal.data_types.core import Model
from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset
from fnp.fincausal.data_types.features import IsCausalConnectivePresentFeature
from fnp.fincausal.data_types.label import FinCausalTask1Pred
from fnp.fincausal.data_types.modeling_instance import FinCausalTask1ModelingInstance
from simpletransformers.classification import ClassificationModel


import numpy as np
import pandas as pd
import sklearn


class SimpleTransformersModel(Model):

    # todo check if sending the args from main script is the right way to do it?
    def __init__(self,
                 model_type: str,
                 model_name_or_path: Union[str, Path],
                 output_dir: Path,
                 class_weights: Optional[List[float]]=None
                 ):
        print('class weights: {}'.format(class_weights))
        self.output_dir = output_dir
        self.cache_dir = output_dir / 'cache/' # todo this should be a property of the data
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
                                               'best_model_dir': str(self.best_model_dir),
                                               'use_early_stopping': True,
                                               'early_stopping_patience': 3},
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
                         train_batch_size: Optional[int] = 8):
        train_df = SimpleTransformersModel._convert_modeling_dataset_to_training_dataset(train_dataset)
        eval_df = SimpleTransformersModel._convert_modeling_dataset_to_training_dataset(val_dataset)
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
                                   'early_stopping_metric_minimize': False
                               },
                               f1=sklearn.metrics.f1_score)

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
        preds, _ = self.model.predict(to_predict=to_predict_texts)
        for modeling_instance, pred in zip(dataset, preds):
            modeling_instance.pred = FinCausalTask1Pred(causal=pred) # ToDo change the conf. score


class FinCausalTask1FindCausalWords(Model):

    def fit_and_evaluate(self, train_dataset: FinCausalTask1ModelingDataset, val_dataset: FinCausalTask1ModelingDataset):
        pass

    def __init__(self):
        """
        self.causal_connectives = ['cause', 'lead to', 'bring about', 'generate', 'make', 'force', 'allow', 'contribute',
                                   'active', 'alert', 'influence', 'provide', 'reduce', 'relax', 'result in',
                                   'results in', 'result', 'increase', 'trigger', 'persist', 'because', 'as', 'since',
                                   'so', 'so that', 'once', 'for this reason', 'with the result that', 'hence',
                                   'therefore', 'consequently', 'following', 'because of', 'thanks to', 'due to',
                                   'as a consequence', 'as a result of']
        """
        self.causal_connectives = ['since', 'as', 'because', 'after']

    def fit(self, dataset: FinCausalTask1ModelingDataset):
        """

        :param dataset:
        :return:
        """

        pass

    def predict_on_dataset(self, dataset: FinCausalTask1ModelingDataset):
        """

        :param dataset:
        :return:
        """
        for modeling_instance in dataset.instances:
            text = modeling_instance.dataset_instance.text.lower()
            modeling_instance.pred = FinCausalTask1Pred(causal=int(any([causal_connective in text for causal_connective in self.causal_connectives])),
                                                        confidence=1.0)

    def predict_on_instance(self, instance: FinCausalTask1ModelingInstance):
        """

        :param instance:
        :return:
        """
        pass


class FinCausalTask1SklearnLogRegModel(Model):

    def __init__(self):
        self.model = LogisticRegression(random_state=42)

    def fit(self, dataset: FinCausalTask1ModelingDataset):
        """
        This function should do everything to train the model. In case of SkLearn models, it is a simple fit.
        In case of a deep learning model, this should run a batcher and run epochs. Ideally, the modeling_instances
        should be converted to the training_instances before passing to the model. Since every model has a different
        input requirement, it makes sense to let this function convert the modeling_instance to the training_instance
        iself.
        :param dataset:
        :return:
        """

        # convert the modeling_instances into input suitable for LogReg model
        X, y = FinCausalTask1SklearnLogRegModel._convert_modeling_dataset_to_training_dataset(dataset)
        print('fitting')
        self.model.fit(X, y)

    @staticmethod
    def _convert_modeling_dataset_to_training_dataset(modeling_dataset: FinCausalTask1ModelingDataset) -> Tuple[np.array, np.array]:
        features_for_each_instance = []
        label_for_each_instance = []
        for modeling_instance in modeling_dataset.instances:
            features_for_each_instance.append(FinCausalTask1SklearnLogRegModel._convert_modeling_instance_to_training_instance(modeling_instance))
            label_for_each_instance.append(FinCausalTask1SklearnLogRegModel._extract_label_for_modeling_instance_(modeling_instance))
        return np.array(features_for_each_instance), np.array(label_for_each_instance)

    @staticmethod
    def _convert_modeling_instance_to_training_instance(modeling_instance: FinCausalTask1ModelingInstance) -> List:

        features = []
        if modeling_instance.features:
            for feature in modeling_instance.features:
                if isinstance(feature, IsCausalConnectivePresentFeature):
                    features.append(1 if feature.is_causal_connective_present else 0)

        return features

    @staticmethod
    def _extract_label_for_modeling_instance_(modeling_instance: FinCausalTask1ModelingInstance) -> int:
        return modeling_instance.label.causal
