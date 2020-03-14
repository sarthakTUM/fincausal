from fnp.fincausal.data_types.core import TextPreProcessor


class LowerCaseTextPreProcessor(TextPreProcessor):
    def __call__(self, text: str) -> str:
        return text.lower()
