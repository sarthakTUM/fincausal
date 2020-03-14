from typing import Tuple, List

from fnp.fincausal.data_types.core import Feature


class IsCausalConnectivePresentFeature(Feature):

    def __init__(self, is_causal_connective_present: bool):
        super(IsCausalConnectivePresentFeature, self).__init__()
        self.is_causal_connective_present = is_causal_connective_present


class POSFeature(Feature):
    def __init__(self, pos_tags_of_each_word_in_each_sentence: List[List[Tuple[str, str]]]):
        super(POSFeature, self).__init__()
        self.pos_tags_of_each_word_in_each_sentence = pos_tags_of_each_word_in_each_sentence
