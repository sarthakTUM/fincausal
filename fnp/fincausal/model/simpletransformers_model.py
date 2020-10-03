from pathlib import Path
from typing import Union, Optional, List
from imblearn.over_sampling import RandomOverSampler
from simpletransformers.classification import ClassificationModel
from fnp.fincausal import utils
from fnp.fincausal.data_types.core import Model
from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset
import pandas as pd
from fnp.fincausal.data_types.label import FinCausalTask1Pred
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
