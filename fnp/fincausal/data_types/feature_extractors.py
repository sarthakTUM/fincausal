from typing import Optional, List, Tuple

from nltk import sent_tokenize, word_tokenize, pos_tag

from fnp.fincausal.data_types.core import FeatureExtractor
from fnp.fincausal.data_types.dataset_instance import FinCausalDatasetInstance
from fnp.fincausal.data_types.features import IsCausalConnectivePresentFeature, POSFeature


class IsCausalConnectivePresentFeatureExtractor(FeatureExtractor):
    def __init__(self):

        # ToDo fill causal connectives
        self.causal_connectives = []

    def extract(self, dataset_instance: FinCausalDatasetInstance) -> IsCausalConnectivePresentFeature:
        """
        extracts from the text a boolean, whether the text contains a causal connective.
        :param dataset_instance:
        :return:
        """
        return IsCausalConnectivePresentFeature(is_causal_connective_present=self._contains_causal_connective(dataset_instance.text))

    def _contains_causal_connective(self, text: str) -> bool:
        """
        checks for causal connective in the text
        :param text:
        :return:
        """
        return any([causal_connective in text for causal_connective in self.causal_connectives])


class POSFeatureExtractor(FeatureExtractor):
    def __init__(self, lang: Optional[str] = 'english'):
        self.lang = lang

    def extract(self, dataset_instance: FinCausalDatasetInstance) -> POSFeature:
        text = dataset_instance.text
        sent_tokenized: List[str] = sent_tokenize(text, self.lang)
        words_tokenized: List[List[str]] = [word_tokenize(sent) for sent in sent_tokenized]
        pos_tags: List[List[Tuple[str, str]]] = [pos_tag(word_tokenized) for word_tokenized in words_tokenized]
        return POSFeature(pos_tags_of_each_word_in_each_sentence=pos_tags)
