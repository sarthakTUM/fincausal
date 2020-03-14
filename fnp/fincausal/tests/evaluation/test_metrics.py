import unittest

from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset
from fnp.fincausal.data_types.dataset_instance import FinCausalTask1DatasetInstance
from fnp.fincausal.data_types.label import FinCausalTask1Label, FinCausalTask1Pred
from fnp.fincausal.data_types.modeling_instance import FinCausalTask1ModelingInstance
from fnp.fincausal.evaluation.metrics import Metrics, ConfMatrixCalculator
from sklearn.metrics import f1_score, confusion_matrix


class TestMetrics(unittest.TestCase):

    def test_f_1_is_equal_to_sklearn_fbeta(self):
        """
        tests that f_beta returned by custom function is equal to sklearn's F1
        """

        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 1, 1]
        tp = 3
        fp = 1
        fn = 1
        tn = 1
        beta = 1.0

        f_beta_custom = Metrics.f_beta(tp=tp, fp=fp, fn=fn, beta=beta)
        f_beta_sklearn = f1_score(y_true=y_true, y_pred=y_pred)

        self.assertEqual(f_beta_custom, f_beta_sklearn)


class TestConfMatrixCalculator(unittest.TestCase):

    def test_return_from_calculate_tp_fp_fn_task1_equal_to_sklearn_confmatrix(self):
        """
        Test that the tp, fp, tn and fn returned by custom conf matrix are equal to that returned by sklearn
        """

        # output from sklearn
        y_true = [1, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 1, 1]

        # output from custom function
        sk_tn, sk_fp, sk_fn, sk_tp = confusion_matrix(y_true, y_pred).ravel()
        fincausal_task1_modeling_dataset = FinCausalTask1ModelingDataset(instances=[
            FinCausalTask1ModelingInstance(dataset_instance=FinCausalTask1DatasetInstance(index=0.1,
                                                                                          text='dataset_instance_1',
                                                                                          gold=1),
                                           features=[],
                                           label=FinCausalTask1Label(causal=1),
                                           pred=FinCausalTask1Pred(causal=0)
                                           ),
            FinCausalTask1ModelingInstance(dataset_instance=FinCausalTask1DatasetInstance(index=0.2,
                                                                                          text='dataset_instance_2',
                                                                                          gold=0),
                                           features=[],
                                           label=FinCausalTask1Label(causal=0),
                                           pred=FinCausalTask1Pred(causal=0)
                                           ),
            FinCausalTask1ModelingInstance(dataset_instance=FinCausalTask1DatasetInstance(index=0.3,
                                                                                          text='dataset_instance_3',
                                                                                          gold=1),
                                           features=[],
                                           label=FinCausalTask1Label(causal=1),
                                           pred=FinCausalTask1Pred(causal=1)
                                           ),
            FinCausalTask1ModelingInstance(dataset_instance=FinCausalTask1DatasetInstance(index=0.4,
                                                                                          text='dataset_instance_4',
                                                                                          gold=1),
                                           features=[],
                                           label=FinCausalTask1Label(causal=1),
                                           pred=FinCausalTask1Pred(causal=1)
                                           ),
            FinCausalTask1ModelingInstance(dataset_instance=FinCausalTask1DatasetInstance(index=0.5,
                                                                                          text='dataset_instance_5',
                                                                                          gold=0),
                                           features=[],
                                           label=FinCausalTask1Label(causal=0),
                                           pred=FinCausalTask1Pred(causal=1)
                                           ),
            FinCausalTask1ModelingInstance(dataset_instance=FinCausalTask1DatasetInstance(index=0.6,
                                                                                          text='dataset_instance_6',
                                                                                          gold=1),
                                           features=[],
                                           label=FinCausalTask1Label(causal=1),
                                           pred=FinCausalTask1Pred(causal=1)
                                           ),
        ])
        cu_tp, cu_fp, cu_fn, cu_tn = ConfMatrixCalculator.calculate_tp_fp_fn_task1(dataset=fincausal_task1_modeling_dataset)

        self.assertEqual(sk_tp, cu_tp)
        self.assertEqual(sk_fp, cu_fp)
        self.assertEqual(sk_fn, cu_fn)
        self.assertEqual(sk_tn, cu_tn)

