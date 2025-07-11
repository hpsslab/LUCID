from typing import Literal

ENTITIES : Literal = Literal[
        "PROBLEM",
        "APPROACH",
        "MATERIAL",
        "PROCESS",
        "METRIC",
        "ADVANTAGE"
]

RELATIONS : Literal = Literal[
        "USED_IN",
        "SOLVES",
        "EVALUATES",
        "OFFERS"
]

VALIDATION_SCHEMA : list(tuple[str, str, str]) = [
        ("MATERIAL", "USED_IN", "APPROACH"),
        ("PROCESS", "USED_IN", "APPROACH"),
        ("APPROACH", "SOLVES", "PROBLEM"),
        ("APPROACH", "EVALUATES", "METRIC"),
        ("APPROACH", "OFFERS", "ADVANTAGE")
]
