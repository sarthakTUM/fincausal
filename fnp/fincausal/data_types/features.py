from typing import Tuple, List

from fnp.fincausal.data_types.core import Feature


class BooleanFeature(Feature):

    def __init__(self, boolean_value: bool):
        super().__init__()
        self.boolean_value = boolean_value


class POSFeature(Feature):
    def __init__(self, pos_tags_of_each_word_in_each_sentence: List[List[Tuple[str, str]]]):
        super(POSFeature, self).__init__()
        self.pos_tags_of_each_word_in_each_sentence = pos_tags_of_each_word_in_each_sentence


class OneHotFeature(Feature):
    def __init__(self, one_hot: List[int]):
        super(OneHotFeature).__init__()
        self.one_hot = one_hot

