from typing import List, Dict

from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset
from sklearn import metrics


class MetricsWrapper:

    @staticmethod
    def calculate_metrics_task1(dataset: FinCausalTask1ModelingDataset) -> Dict[str, float]:
        """
        calculates the metrics for FinCausal Taks 1
        """
        true = []
        preds = []

        for modeling_instance in dataset.instances:
            true.append(modeling_instance.label.causal)
            preds.append(modeling_instance.pred.causal)

        return Metrics.task1(true, preds)

    @staticmethod
    def calculate_confusionmatrix_task1(dataset: FinCausalTask1ModelingDataset) -> Dict[str, int]:
        """
        calculates TP, FP, FN and TN for FinCausal Task 1

        """

        true = []
        preds = []

        for modeling_instance in dataset.instances:
            true.append(modeling_instance.label.causal)
            preds.append(modeling_instance.pred.causal)

        return Metrics.binary_confusion_matrix(true, preds)


class Metrics:

    @staticmethod
    def f_beta(tp: int, fp: int, fn: int, beta: float) -> float:
        beta_value = (1+(beta**2))
        precision = Metrics.precision(tp, fp)
        recall = Metrics.recall(tp, fn)
        f_beta_num = beta_value * (precision * recall)
        f_beta_den = ((beta**2) * precision) + recall
        return f_beta_num/f_beta_den

    @staticmethod
    def precision(tp: int, fp: int) -> float:
        return (tp)/(tp+fp)

    @staticmethod
    def recall(tp: int, fn: int) -> float:
        return (tp)/(tp+fn)

    @staticmethod
    def weighted_f1_score(y_true: List[int], y_pred: List[int]) -> float:
        return metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    @staticmethod
    def task1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_true,
                                                                                          y_pred,
                                                                                          beta=1.0,
                                                                                          average='weighted')
        return {
            'precision': precision,
            'recall': recall,
            'f1': fbeta_score,
            'support': support
        }

    @staticmethod
    def binary_confusion_matrix(y_true: List[int], y_pred: List[int]):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }

