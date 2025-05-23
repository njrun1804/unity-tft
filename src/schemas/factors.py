from typing import TypedDict, Dict

class FactorSchema(TypedDict):
    run_timestamp: str
    ticker: str
    features: Dict[str, float]
