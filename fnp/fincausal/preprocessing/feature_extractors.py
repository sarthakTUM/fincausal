from typing import Optional, List, Tuple

import spacy
from nltk import sent_tokenize, word_tokenize, pos_tag
from fnp.fincausal.data_types.core import FeatureExtractor
from fnp.fincausal.data_types.dataset_instance import FinCausalDatasetInstance
from fnp.fincausal.data_types.features import POSFeature, BooleanFeature, OneHotFeature

import re


class SubstringPresentFeatureExtractor(FeatureExtractor):

    def __init__(self, substrings: List[str]):
        self.substrings = substrings

    def extract(self, dataset_instance: FinCausalDatasetInstance) -> BooleanFeature:
        return BooleanFeature(boolean_value=any(substring in dataset_instance.text for substring in self.substrings))


class RegexPresentFeatureExtractor(FeatureExtractor):

    def __init__(self, regex: str):
        self.regex = regex

    def extract(self, dataset_instance: FinCausalDatasetInstance) -> BooleanFeature:
        return BooleanFeature(boolean_value=True if len(re.findall(self.regex, dataset_instance.text)) >0 else False)


class ContainsCausalConnectiveFeatureExtractor(SubstringPresentFeatureExtractor):

    def __init__(self, causal_connectives: Optional[List[str]] = None):

        self.causal_connectives = causal_connectives if causal_connectives else []
        super().__init__(substrings=self.causal_connectives)


class POSFeatureExtractor(FeatureExtractor):
    def __init__(self, lang: Optional[str] = 'english'):
        self.lang = lang

    def extract(self, dataset_instance: FinCausalDatasetInstance) -> POSFeature:
        text = dataset_instance.text
        sent_tokenized: List[str] = sent_tokenize(text, self.lang)
        words_tokenized: List[List[str]] = [word_tokenize(sent) for sent in sent_tokenized]
        pos_tags: List[List[Tuple[str, str]]] = [pos_tag(word_tokenized) for word_tokenized in words_tokenized]
        return POSFeature(pos_tags_of_each_word_in_each_sentence=pos_tags)


class ContainsNumericFeatureExtractor(FeatureExtractor):

    def extract(self, dataset_instance: FinCausalDatasetInstance) -> BooleanFeature:
        num_digits = sum([char.isdigit() for char in dataset_instance.text])
        return BooleanFeature(boolean_value=num_digits != 0)


class ContainsPercentFeatureExtractor(SubstringPresentFeatureExtractor):

    def __init__(self):
        super(ContainsPercentFeatureExtractor, self).__init__(substrings=['%'])


class ContainsCurrencyFeatureExtractor(SubstringPresentFeatureExtractor):

    def __init__(self, currencies: Optional[List[str]]):
        super(ContainsCurrencyFeatureExtractor, self).__init__(substrings=currencies)


class ContainsSpecificVerbAfterCommaFeatureExtractor(SubstringPresentFeatureExtractor):

    def __init__(self, verbs: Optional[List[str]] = None):
        self.verbs = verbs if verbs else []
        super().__init__(substrings=self.verbs)


class ContainsTextualNumericFeatureExtractor(SubstringPresentFeatureExtractor):

    def __init__(self, textual_numerics: Optional[List[str]]=None):
        self.textual_numerics = textual_numerics if textual_numerics else []
        super(ContainsTextualNumericFeatureExtractor, self).__init__(substrings=self.textual_numerics)


class ContainsVerbAfterCommaFeatureExtractor(RegexPresentFeatureExtractor):

    def __init__(self, regex: str):
        self.regex = regex
        super().__init__(regex=regex)


class POSofRootFeatureExtractor(FeatureExtractor):

    def extract(self, dataset_instance: FinCausalDatasetInstance) -> OneHotFeature:
        one_hot = [0]*len(self.pos_cats)
        doc = self.nlp(dataset_instance.text)
        for token in doc:
            if token.dep_ == 'ROOT':
                one_hot[self.pos_cats.index(token.pos_.upper())] = 1
        return OneHotFeature(one_hot=one_hot)

    def \
            __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.pos_cats = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                         'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE']





