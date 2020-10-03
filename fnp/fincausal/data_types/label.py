from typing import Optional
from fnp.fincausal.data_types.core import FinCausalLabel


class FinCausalTask1Label(FinCausalLabel):
    def __init__(self, causal: int):
        super().__init__()
        self.causal = causal


class FinCausalTask1Pred(FinCausalTask1Label):
    def __init__(self, causal: int, confidence: Optional[float] = None):
        super().__init__(causal=causal)
        self.confidence = confidence
