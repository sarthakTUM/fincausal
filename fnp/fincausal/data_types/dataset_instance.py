from typing import Optional
from fnp.fincausal.data_types.core import FinCausalDatasetInstance


class FinCausalTask1DatasetInstance(FinCausalDatasetInstance):
    """
    Data Model for instance of task 1
    """
    def __init__(self,
                 unique_id: str,
                 index: str,
                 text: str,
                 gold: Optional[int] = None):
        """

        :param unique_id: custom generated ID (uuid)
        :param index: ID provided by FinCausal
        :param text: raw text segment
        :param gold: gold label by FinCausal
        """
        super().__init__(unique_id=unique_id, index=index, text=text)
        self.gold = gold


class FinCausalTask2DatasetInstance(FinCausalDatasetInstance):
    """
    This class is not used in this code because we are not working in Task 2
    """
    def __init__(self,
                 index: str,
                 text: str,
                 cause: str,
                 effect: str,
                 cause_start: int,
                 cause_end: int,
                 effect_start: int,
                 effect_end: int,
                 sentence: str
                 ):
        super().__init__(index=index, text=text)
        self.cause = cause
        self.effect: str = effect
        self.cause_start: int = cause_start
        self.cause_end: int = cause_end
        self.effect_start: int = effect_start
        self.effect_end: int = effect_end
        self.sentence: str = sentence
