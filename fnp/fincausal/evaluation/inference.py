from pathlib import Path
from typing import Optional
import pandas as pd
from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset


class FinCausalTask1Inference:
    """
    Class for holding the inference values for the FinCausal Task 1
    """

    def __init__(self,
                 model_path: Path,
                 predict_file_path: Path,
                 predict_modeling_dataset: FinCausalTask1ModelingDataset,
                 output_dir: Path,
                 train_file_path: Optional[Path] = None,
                 val_file_path: Optional[Path] = None,
                 f1: Optional[float] = None,
                 recall: Optional[float] = None,
                 precision: Optional[float] = None,
                 tp: Optional[int] = None,
                 fp: Optional[int] = None,
                 fn: Optional[int] = None,
                 tn: Optional[int] = None
                 ):
        """
        :param model_path: path to model
        :param predict_file_path: path to final predictions
        :param predict_modeling_dataset: dataset to predict on
        :param output_dir: output path where training artifacts are stored
        :param train_file_path: path to the training file used
        :param val_file_path: path to the validation file used
        :param f1: F1 score
        :param recall: recall score
        :param precision: precision score
        :param tp: True positives
        :param fp: False Positives
        :param fn: False Negatives
        :param tn: True Negatives
        """
        self.model_path = model_path
        self.predict_file_path = predict_file_path
        self.f1 = f1
        self.recall = recall
        self.precision = precision
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self.predict_modeling_dataset = predict_modeling_dataset
        self.output_dir = output_dir
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path

        if self.output_dir.exists():
            print('{} exists. Delete the folder'.format(self.output_dir))
            raise FileExistsError

        self.output_dir.mkdir(parents=False, exist_ok=False)

        self._write_paths()
        self.write_metrics_to_file()
        self.write_predictions_to_csv()

    def write_metrics_to_file(self, output_dir: Optional[Path]=None):

        """
        write the metrics to the specified filepath in a text file
        :param output_dir: force to write to a new output_dir
        """

        output_dir = output_dir if output_dir else self.output_dir

        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write('f1: {}\n'.format(self.f1))
            f.write('precision: {}\n'.format(self.precision))
            f.write('recall: {}\n'.format(self.recall))
            f.write('tp: {}\n'.format(self.tp))
            f.write('fp: {}\n'.format(self.fp))
            f.write('fn: {}\n'.format(self.fn))
            f.write('tn: {}\n'.format(self.tn))

    def write_predictions_to_csv(self, output_dir: Optional[Path]=None):

        """
        write the predictions from the test_modeling_dataset to a csv file
        :param output_dir: force to write to a new output_dir
        """

        output_dir = output_dir if output_dir else self.output_dir
        preds = pd.DataFrame({
            'Index': [modeling_instance.dataset_instance.index for modeling_instance in self.predict_modeling_dataset.instances],
            'Text': [modeling_instance.dataset_instance.text for modeling_instance in self.predict_modeling_dataset.instances],
            'Gold': [modeling_instance.label.causal for modeling_instance in self.predict_modeling_dataset.instances],
            'Prediction': [modeling_instance.pred.causal for modeling_instance in self.predict_modeling_dataset.instances],
            'unique_id': [modeling_instance.dataset_instance.unique_id for modeling_instance in self.predict_modeling_dataset.instances],
            'softmax': [modeling_instance.pred.confidence for modeling_instance in self.predict_modeling_dataset.instances]
        })
        preds.to_csv(output_dir / 'predictions.csv', index=False)

    def _write_paths(self):
        with open(Path(self.output_dir / 'paths.txt'), 'w') as f:
            f.write('model path: {}\n'.format(str(self.model_path)))
            f.write('train path: {}\n'.format(str(self.train_file_path)))
            f.write('val path: {}\n'.format(str(self.val_file_path)))
            f.write('predict path: {}\n'.format(str(self.predict_file_path)))
