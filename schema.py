import typing_extensions as typing

# response schema
class Schema(typing.TypedDict):
    problem: str
    approach: str
    materials: list[str]
    processes: list[str]
    metrics: list[str]
    advantages: list[str]

# relations between labels as defined in the ontology
relations = [
    "[MATERIALS] are used in this [APPROACH]",
    "[PROCESSES] are used in this [APPROACH]",
    "[APPROACH] solves [PROBLEM]",
    "[APPROACH] evaluates [METRICS]",
    "[APPROACH] offers [ADVANTAGES]"
]

