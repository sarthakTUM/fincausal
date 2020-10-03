from typing import Tuple, List
from fnp.fincausal.data_types.core import Feature


class BooleanFeature(Feature):
    """
    feature with a Bool value
    """
    def __init__(self, boolean_value: bool):
        super().__init__()
        self.boolean_value = boolean_value


class POSFeature(Feature):
    """
    Feature to hold POS tag for each word
    """
    def __init__(self, pos_tags_of_each_word_in_each_sentence: List[List[Tuple[str, str]]]):
        super(POSFeature, self).__init__()
        self.pos_tags_of_each_word_in_each_sentence = pos_tags_of_each_word_in_each_sentence


class OneHotFeature(Feature):
    """
    feature for one-hot vector
    """
    def __init__(self, one_hot: List[int]):
        super(OneHotFeature).__init__()
        self.one_hot = one_hot

