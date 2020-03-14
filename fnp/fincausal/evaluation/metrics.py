from fnp.fincausal.data_types.dataset import FinCausalTask1ModelingDataset


class ConfMatrixCalculator:

    @staticmethod
    def calculate_tp_fp_fn_task1(dataset: FinCausalTask1ModelingDataset) -> (int, int, int, int):

        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for modeling_instance in dataset.instances:
            if modeling_instance.label.causal == 1 and modeling_instance.pred.causal == 1:
                tp += 1
            if modeling_instance.label.causal == 1 and modeling_instance.pred.causal == 0:
                fn += 1
            if modeling_instance.label.causal == 0 and modeling_instance.pred.causal == 1:
                fp += 1
            if modeling_instance.label.causal == 0 and modeling_instance.pred.causal == 0:
                tn += 1
        return tp, fp, fn, tn

    @staticmethod
    def calculate_tp_fp_fn_task2():
        pass


class Metrics:

    @staticmethod
    def f_beta(tp: int, fp: int, fn: int, beta: float) -> float:
        # todo handle division by zero
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


